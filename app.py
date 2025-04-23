from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import uuid
import cv2
import tempfile
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import base64
import shutil
from typing import List, Optional
from pydantic import BaseModel
import io

app = FastAPI()

# デプロイ環境では環境変数が直接設定されるため、load_dotenv()は開発環境でのみ使用
if os.getenv("AZURE_ENVIRONMENT") != "production":
    load_dotenv()

# CORSミドルウェアの設定（修正：すべてのオリジンを許可）
app.add_middleware(
    CORSMiddleware,
    # 修正：許可するオリジンを追加
    allow_origins=[
        "https://tech0-gen-8-step4-peak-front-d2ayf3aca3e3bcdu.canadacentral-01.azurewebsites.net",
        # 開発環境用
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "https://tech0-gen-8-step4-peak-front-d2ayf3aca3e3bcdu.z01.azurefd.net"
    ],
    # クレデンシャルを含むリクエストも許可
    allow_credentials=True,
    # すべてのHTTPメソッドを許可
    allow_methods=["*"],
    # すべてのヘッダーを許可
    allow_headers=["*"],
    # プリフライトリクエストの結果をキャッシュする時間（秒）
    max_age=3600,
)

# Azure Blob Storageの接続情報
blob_connection_string = os.getenv('BLOB_CONNECTION_STRING')
container_name = os.getenv('BLOB_CONTAINER_NAME')

# BlobServiceClientの初期化
blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
container_client = blob_service_client.get_container_client(container_name)

# データベース接続情報
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

# SSL証明書の取得
SSL_CA_CERT = os.getenv("SSL_CA_CERT")
if not SSL_CA_CERT:
    raise ValueError(":x: SSL_CA_CERT が設定されていません！")

# SSL証明書の一時ファイル作成
def create_ssl_cert_tempfile():
    pem_content = SSL_CA_CERT.replace("\\n", "\n").replace("\\", "")
    temp_pem = tempfile.NamedTemporaryFile(delete=False, suffix=".pem", mode="w")
    temp_pem.write(pem_content)
    temp_pem.close()
    return temp_pem.name

SSL_CA_PATH = create_ssl_cert_tempfile()

# Azure環境ではApp Serviceのファイルシステムは一時的なもの
# 以下のディレクトリはローカルでの処理にのみ使用
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/masks", exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/results", exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/materials", exist_ok=True)

# 静的ファイル配信設定（開発環境のみ）
if os.getenv("AZURE_ENVIRONMENT") != "production":
    app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# リクエスト/レスポンスモデル
class ProcessRequest(BaseModel):
    image_id: str
    mask_data: str  # base64エンコードされたマスクデータ

class MaterialRequest(BaseModel):
    image_id: str
    mask_id: str
    material_id: str

# Blobヘルパー関数
async def save_to_blob_storage(file_content, blob_name, content_type=None):
    """ファイルをBlobストレージに保存し、URLを返す"""
    try:
        # Blobクライアントの取得
        blob_client = container_client.get_blob_client(blob_name)
        
        # アップロード
        blob_client.upload_blob(file_content, overwrite=True, content_settings=None)
        
        # URLを返す
        return blob_client.url
    except Exception as e:
        print(f"Blobストレージエラー: {e}")
        raise

async def get_blob_content(blob_name):
    """Blobからコンテンツを取得"""
    try:
        blob_client = container_client.get_blob_client(blob_name)
        download_stream = blob_client.download_blob()
        return await download_stream.readall()
    except Exception as e:
        print(f"Blob取得エラー: {e}")
        raise

# データベース接続関数
def get_db_connection():
    """データベース接続を取得"""
    connection = None
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            ssl_ca=SSL_CA_PATH
        )
        return connection
    except Error as e:
        print(f"データベース接続エラー: {e}")
        if connection and connection.is_connected():
            connection.close()
        raise HTTPException(status_code=500, detail=f"データベース接続エラー: {str(e)}")

# API: データベース情報取得
@app.get("/api/db-info")
async def get_db_info():
    """
    データベース情報を取得するAPI（開発用）
    """
    # 本番環境では無効化
    if os.getenv("AZURE_ENVIRONMENT") == "production":
        raise HTTPException(status_code=403, detail="この操作は開発環境でのみ許可されています")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # テーブル一覧を取得
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        table_names = [list(table.values())[0] for table in tables]
        
        result = {
            "tables": table_names,
            "table_details": {}
        }
        
        # 各テーブルの情報を取得
        for table_name in table_names:
            # テーブルのカラム情報を取得
            cursor.execute(f"DESCRIBE {table_name}")
            columns = cursor.fetchall()
            column_names = [col['Field'] for col in columns]
            
            # まずカラム一覧からプライマリキーを特定
            primary_key = None
            for col in columns:
                if col['Key'] == 'PRI':
                    primary_key = col['Field']
                    break
            
            # 最新の1件を取得（プライマリキーがあればそれでソート、なければソートなし）
            if primary_key:
                cursor.execute(f"SELECT * FROM {table_name} ORDER BY {primary_key} DESC LIMIT 1")
            else:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
            
            latest_record = cursor.fetchone()
            
            # 結果を保存
            result["table_details"][table_name] = {
                "columns": column_names,
                "latest_record": latest_record if latest_record else "データなし"
            }
        
        cursor.close()
        conn.close()
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"データベース情報取得エラー: {str(e)}")

# API: 画像アップロード
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # ファイル名の生成
        filename = f"{uuid.uuid4()}.{file.filename.split('.')[-1]}"
        
        # ファイルを読み込む
        file_content = await file.read()
        
        # BlobStorageに直接アップロード
        blob_url = await save_to_blob_storage(file_content, filename)
        
        # データベースに情報を保存
        connection = get_db_connection()
        cursor = connection.cursor()
        try:
            query = """
                INSERT INTO upload_images (filename, blob_url, upload_date)
                VALUES (%s, %s, %s)
            """
            # 日本標準時で保存
            jst = ZoneInfo("Asia/Tokyo")
            upload_date = datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(query, (filename, blob_url, upload_date))
            connection.commit()
        finally:
            cursor.close()
            connection.close()
            
        return {
            "message": "画像が正常にアップロードされました", 
            "filename": filename,
            "blob_url": blob_url
        }
    except Exception as e:
        print(f"アップロードエラー: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# 画像処理ヘルパー関数
def process_mask_image(mask_data_base64, output_path):
    """Base64マスクデータを処理して保存"""
    print("==== process_mask_image関数開始 ====")
    try:
        # Base64から画像データを取得
        if ',' in mask_data_base64:
            print("Base64データにカンマが含まれています、分割処理します")
            mask_data_base64 = mask_data_base64.split(',')[1]
        
        print(f"Base64データ先頭部分: {mask_data_base64[:30]}...")
        mask_data = base64.b64decode(mask_data_base64)
        print(f"デコード後のデータサイズ: {len(mask_data)} バイト")
        
        # バイナリデータからNumpyアレイに変換
        mask_arr = np.frombuffer(mask_data, np.uint8)
        print(f"Numpyアレイ形状: {mask_arr.shape}")
        
        mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("cv2.imdecodeがNoneを返しました")
            return False
            
        print(f"マスク画像サイズ: {mask.shape}")
        
        # 二値化処理
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        print("二値化処理完了")
        
        # 処理済みマスクを保存
        cv2.imwrite(output_path, binary_mask)
        print(f"マスク保存完了: {output_path}")
        
        print("==== process_mask_image関数終了: 成功 ====")
        return True
    except Exception as e:
        print(f"マスク処理エラー: {e}")
        print("==== process_mask_image関数終了: 失敗 ====")
        return False

# API: マスク処理
@app.post("/api/process")
async def process_mask(request: ProcessRequest):
    """ユーザーが描いたマスク領域を処理"""
    print(f"===== マスク処理開始: image_id {request.image_id} =====")
    
    connection = None
    cursor = None
    try:
        # 元画像の情報を取得
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute(
            "SELECT * FROM upload_images WHERE upload_id = %s OR filename = %s", 
            (request.image_id, request.image_id)
        )
        image_info = cursor.fetchone()
        
        if not image_info:
            raise HTTPException(status_code=404, detail="画像が見つかりません")
        
        # マスク処理
        mask_id = str(uuid.uuid4())
        mask_filename = f"{mask_id}.png"
        mask_path = f"{UPLOAD_DIR}/masks/{mask_filename}"
        
        # マスクデータを処理して一時ファイルに保存
        if not process_mask_image(request.mask_data, mask_path):
            raise HTTPException(status_code=500, detail="マスク処理に失敗しました")
        
        # 一時ファイルをBlobストレージにアップロード
        with open(mask_path, "rb") as f:
            mask_blob_url = await save_to_blob_storage(f.read(), f"masks/{mask_filename}")
        
        # データベースにマスク情報を保存 - mask_pathカラムにBlobのURLを保存
        cursor.execute(
            "INSERT INTO masks (mask_id, image_id, mask_path, created_at) VALUES (%s, %s, %s, NOW())",
            (mask_id, image_info["filename"], mask_blob_url)  # mask_pathにBlobのURLを保存
        )
        connection.commit()
        
        # Azureでのパブリックアクセス用URL
        public_url = mask_blob_url
        
        return {
            "success": True,
            "image_id": image_info["filename"],
            "mask_id": mask_id,
            "mask_path": mask_path,
            "public_url": public_url
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"マスク処理エラー: {e}")
        raise HTTPException(status_code=500, detail=f"マスク処理エラー: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

# 素材を適用する処理
def apply_material_to_image(original_path, mask_path, material_path, output_path):
    """
    マスク領域に素材を適用
    """
    print(f"==== 素材適用処理開始 ====")
    print(f"元画像: {original_path}")
    print(f"マスク: {mask_path}")
    print(f"素材: {material_path}")
    print(f"出力先: {output_path}")
    
    try:
        # 画像読み込み
        print("画像読み込み中...")
        original = cv2.imread(original_path)
        if original is None:
            print(f"元画像の読み込みに失敗: {original_path}")
            return False
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"マスク画像の読み込みに失敗: {mask_path}")
            return False
        
        material = cv2.imread(material_path)
        if material is None:
            print(f"素材画像の読み込みに失敗: {material_path}")
            return False
        
        print(f"元画像サイズ: {original.shape}")
        print(f"マスクサイズ: {mask.shape}")
        print(f"素材サイズ: {material.shape}")
        
        # サイズ調整
        print("素材画像をリサイズ中...")
        material = cv2.resize(material, (original.shape[1], original.shape[0]))
        print(f"リサイズ後の素材サイズ: {material.shape}")
        
        # マスク処理
        print("マスク状態を確認中...")
        mask_mean = np.mean(mask)
        print(f"マスクの平均輝度値: {mask_mean}")
        
        if np.mean(mask) > 127:  # マスクが主に白の場合
            print("マスクが白い領域を主に含むため反転します")
            mask_inv = cv2.bitwise_not(mask)  # 反転して黒を選択領域にする
        else:
            print("マスクが黒い領域を主に含むため反転しません")
            mask_inv = mask  # すでに黒が選択領域の場合
            mask = cv2.bitwise_not(mask_inv)  # 白を非選択領域にする
        
        # 背景（マスクされていない領域）
        print("背景処理中...")
        background = cv2.bitwise_and(original, original, mask=mask)
        
        # 前景（マスクされた領域に素材を適用）
        print("前景処理中...")
        foreground = cv2.bitwise_and(material, material, mask=mask_inv)
        
        # 合成
        print("画像合成中...")
        result = cv2.add(background, foreground)
        
        # 結果を保存
        print(f"結果を保存中: {output_path}")
        save_success = cv2.imwrite(output_path, result)
        if not save_success:
            print(f"結果の保存に失敗しました: {output_path}")
            return False
        
        print(f"==== 素材適用処理完了: 成功 ====")
        return True
    except Exception as e:
        print(f"素材適用エラー: {e}")
        print(f"==== 素材適用処理完了: 失敗 ====")
        return False

# API: 素材一覧取得
@app.get("/api/materials/{category}")
async def get_materials_by_category(category: str):
    """
    カテゴリ別の素材一覧を取得
    """
    print(f"===== カテゴリ[{category}]の素材一覧取得開始 =====")
    
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # データベースの全カテゴリを確認（デバッグ用）
        cursor.execute("SELECT DISTINCT category FROM products")
        categories = cursor.fetchall()
        print(f"データベースの全カテゴリ: {[cat['category'] for cat in categories]}")
        
        # カテゴリに合致する素材を取得
        query = """
            SELECT pi.image_id, pi.product_id, pi.image_url, 
                   p.category, p.series, p.color, p.price
            FROM products_image pi
            JOIN products p ON pi.product_id = p.product_id
            WHERE p.category = %s
            ORDER BY p.series, p.color
        """
        print(f"実行するSQL: {query} パラメータ: [{category}]")
        cursor.execute(query, (category,))
        
        materials = cursor.fetchall()
        print(f"クエリ結果: {len(materials)}件")
        
        if not materials:
            # カテゴリに関するさらなる情報を取得
            cursor.execute("SELECT COUNT(*) as count FROM products WHERE category = %s", (category,))
            cat_count = cursor.fetchone()
            print(f"category={category}の商品は{cat_count['count']}件存在します")
            
            # products_imageテーブルのチェック
            cursor.execute("SELECT COUNT(*) as count FROM products_image")
            img_count = cursor.fetchone()
            print(f"products_imageテーブルには{img_count['count']}件のレコードが存在します")
            
            print(f"カテゴリ[{category}]の素材が見つかりません")
            return {"materials": []}
        
        # 結果のサンプル出力
        print(f"最初の素材サンプル: {materials[0]}")
        
        # 結果を整形して返す
        result = [
            {
                "id": material["image_id"],
                "product_id": material["product_id"],
                "name": material["series"],
                "color": material["color"],
                "price": material["price"],
                "image": material["image_url"]
            }
            for material in materials
        ]
        
        print(f"===== カテゴリ[{category}]の素材一覧取得完了: {len(result)}件 =====")
        return {"materials": result}
    
    except Exception as e:
        print(f"素材一覧取得エラー: {e}")
        # スタックトレースも出力
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"素材一覧取得エラー: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

# API: 素材適用
@app.post("/api/apply-material")
async def apply_material(request: MaterialRequest):
    """
    選択された素材をマスク領域に適用
    """
    print(f"===== 素材適用API開始 =====")
    print(f"リクエスト: image_id={request.image_id}, mask_id={request.mask_id}, material_id={request.material_id}")
    
    connection = None
    cursor = None
    try:
        # 必要な情報を取得
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # 元画像情報 - upload_idまたはfilenameで検索
        cursor.execute("SELECT * FROM upload_images WHERE upload_id = %s OR filename = %s", 
                      (request.image_id, request.image_id))
        image_info = cursor.fetchone()
        if not image_info:
            raise HTTPException(status_code=404, detail="画像が見つかりません")
        
        # マスク情報
        cursor.execute("SELECT * FROM masks WHERE mask_id = %s", (request.mask_id,))
        mask_info = cursor.fetchone()
        if not mask_info:
            raise HTTPException(status_code=404, detail="マスクが見つかりません")
        
        # 素材情報 - image_idまたはproduct_idで検索
        cursor.execute("""
            SELECT pi.*, p.category, p.series, p.color, p.price 
            FROM products_image pi
            JOIN products p ON pi.product_id = p.product_id
            WHERE pi.image_id = %s OR pi.product_id = %s
        """, (request.material_id, request.material_id))
        material_info = cursor.fetchone()
        if not material_info:
            raise HTTPException(status_code=404, detail="素材が見つかりません")
        
        # 元画像、マスク、素材を一時ディレクトリにダウンロード
        temp_original_path = os.path.join(UPLOAD_DIR, f"temp_original_{uuid.uuid4()}.jpg")
        temp_mask_path = os.path.join(UPLOAD_DIR, f"temp_mask_{uuid.uuid4()}.png")
        temp_material_path = os.path.join(UPLOAD_DIR, f"temp_material_{uuid.uuid4()}.jpg")
        
        # 元画像をダウンロード
        import requests
        response = requests.get(image_info["blob_url"])
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="元画像のダウンロードに失敗しました")
        with open(temp_original_path, "wb") as f:
            f.write(response.content)
        
        # マスクをダウンロード - mask_pathにはBlobのURLが保存されている
        response = requests.get(mask_info["mask_path"])
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="マスクのダウンロードに失敗しました")
        with open(temp_mask_path, "wb") as f:
            f.write(response.content)
        
        # 素材をダウンロード
        response = requests.get(material_info["image_url"])
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="素材のダウンロードに失敗しました")
        with open(temp_material_path, "wb") as f:
            f.write(response.content)
        
        # 合成処理
        result_id = str(uuid.uuid4())
        temp_result_path = os.path.join(UPLOAD_DIR, f"results/{result_id}.jpg")
        
        # 素材を適用
        if not apply_material_to_image(
            temp_original_path, 
            temp_mask_path, 
            temp_material_path, 
            temp_result_path
        ):
            raise HTTPException(status_code=500, detail="素材適用に失敗しました")
        
        # 結果をBlobストレージにアップロード
        with open(temp_result_path, "rb") as f:
            result_blob_url = await save_to_blob_storage(f.read(), f"results/{result_id}.jpg")
        
        # 結果をデータベースに保存 - result_pathにBlobのURLを保存
        cursor.execute(
            """
            INSERT INTO results 
            (result_id, image_id, mask_id, material_id, result_path, created_at) 
            VALUES (%s, %s, %s, %s, %s, NOW())
            """,
            (
                result_id, 
                image_info["filename"],
                request.mask_id, 
                material_info["image_id"],
                result_blob_url  # result_pathにBlobのURLを保存
            )
        )
        connection.commit()
        
        # 一時ファイルの削除
        for path in [temp_original_path, temp_mask_path, temp_material_path]:
            if os.path.exists(path):
                os.remove(path)
        
        return {
            "success": True,
            "result_id": result_id,
            "result_path": temp_result_path,
            "public_url": result_blob_url
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"素材適用APIエラー: {e}")
        raise HTTPException(status_code=500, detail=f"素材適用エラー: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

# API: Before画像取得
@app.get("/api/preview/before/{image_id}")
async def get_before_image(image_id: str):
    """
    元の画像（Before）を取得
    """
    print(f"===== Before画像取得: image_id {image_id} =====")
    
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # upload_idまたはfilenameで検索
        cursor.execute("SELECT * FROM upload_images WHERE upload_id = %s OR filename = %s", 
                      (image_id, image_id))
        image_info = cursor.fetchone()
        
        if not image_info:
            raise HTTPException(status_code=404, detail="画像が見つかりません")
        
        return {
            "image_id": image_info["filename"],
            "blob_url": image_info["blob_url"]
        }
    
    except Exception as e:
        print(f"Before画像取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Before画像取得エラー: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

# API: After画像取得
@app.get("/api/preview/after/{image_id}/{mask_id}")
async def get_after_image(image_id: str, mask_id: str):
    """
    合成後の画像（After）を取得
    """
    print(f"===== After画像取得: image_id {image_id}, mask_id {mask_id} =====")
    
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # resultsテーブルから最新の合成画像を取得
        cursor.execute("""
            SELECT * FROM results 
            WHERE image_id = %s AND mask_id = %s 
            ORDER BY created_at DESC LIMIT 1
        """, (image_id, mask_id))
        
        result_info = cursor.fetchone()
        
        if not result_info:
            print(f"合成画像が見つかりません: image_id={image_id}, mask_id={mask_id}")
            # 画像が見つからない場合、代替として元の画像を取得
            cursor.execute("SELECT * FROM upload_images WHERE filename = %s", (image_id,))
            original_image = cursor.fetchone()
            if original_image:
                print(f"代替として元画像を返します: {original_image['filename']}")
                return {
                    "result_id": "original",
                    "result_path": "",
                    "public_url": original_image["blob_url"]
                }
            raise HTTPException(status_code=404, detail="合成画像が見つかりません")
        
        # result_pathにはAzureでのBlobURLが保存されている
        return {
            "result_id": result_info["result_id"],
            "result_path": result_info["result_path"],
            "public_url": result_info["result_path"]  # result_pathにBlobのURLを保存
        }
    
    except Exception as e:
        print(f"After画像取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"After画像取得エラー: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

# ルートエンドポイント
@app.get("/")
async def root():
    return {"message": "FastAPI is running on Azure!"}

# ローカル開発環境でのみUvicornサーバーを起動
if __name__ == "__main__" and os.getenv("AZURE_ENVIRONMENT") != "production":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
