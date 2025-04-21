from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends #後半3つ安田追加
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles #安田追加
from fastapi.responses import FileResponse #安田追加
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import uuid
import cv2 #安田追加
import tempfile
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np #安田追加
import base64 #安田追加
import shutil #安田追加
from typing import List, Optional #安田追加
from pydantic import BaseModel #安田追加
import io #安田追加




app = FastAPI()

# .env ファイルを読み込む
load_dotenv() ## ローカルではここが必要なのでコメントアウトを外す。合わせて.envも作成（4/4 羽田野）

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"],
    allow_origins=["http://localhost:3000"],  # Next.jsのローカルのデフォルトポート指定（4/4 羽田野）   # これでもいいっぽい allow_origins=[os.getenv("CORS_ORIGINS", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# # SSL証明書の一時ファイル作成
def create_ssl_cert_tempfile():
    pem_content = SSL_CA_CERT.replace("\\n", "\n").replace("\\", "")
    temp_pem = tempfile.NamedTemporaryFile(delete=False, suffix=".pem", mode="w")
    temp_pem.write(pem_content)
    temp_pem.close()
    return temp_pem.name

SSL_CA_PATH = create_ssl_cert_tempfile()

# ------ここから------
# アップロード用ディレクトリを作成
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/masks", exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/results", exist_ok=True)

# 静的ファイル配信設定
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# リクエスト/レスポンスモデル
class ProcessRequest(BaseModel):
    image_id: str
    mask_data: str  # base64エンコードされたマスクデータ

class MaterialRequest(BaseModel):
    image_id: str
    mask_id: str
    material_id: str


    
# データベース接続関数
def get_db_connection():
    """データベース接続を取得"""
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "refotoru_db"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "")
        )
        return connection
    except Error as e:
        print(f"データベース接続エラー: {e}")
        raise HTTPException(status_code=500, detail=f"データベース接続エラー: {str(e)}")  
    


# ▼▼▼以下、/安田がlocalhost:8000/api/db-infoでDBの中身見るための実装用の処理▼▼▼
@app.get("/api/db-info")
async def get_db_info():
    """
    データベース情報を取得するAPI（開発用）
    """
    # if os.getenv("ENVIRONMENT") != "development":
    #     raise HTTPException(status_code=403, detail="この操作は開発環境でのみ許可されています")
    
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
  
    
# ------ここまで安田------



# ▼▼▼以下、/upload画面での処理▼▼▼
def store_image_metadata(filename: str, blob_url: str):
    """アップロードした画像のメタデータをMySQLに保存する"""
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            ssl_ca=SSL_CA_PATH  # SSL CA証明書を使ったセキュア接続
        )
        if connection.is_connected():
            cursor = connection.cursor()
            query = """
                INSERT INTO upload_images (filename, blob_url, upload_date)
                VALUES (%s, %s, %s)
            """
            # 正しい時間で保存するため、日本標準時（Asia/Tokyo）を ZoneInfo で指定
            jst = ZoneInfo("Asia/Tokyo")
            upload_date = datetime.now(jst).strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(query, (filename, blob_url, upload_date))
            connection.commit()
            cursor.close()
        connection.close()
    except Error as e:
        print(f"Database error: {e}")
    finally:
        if connection.is_connected():
            connection.close()

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # 画像ファイルを一時的に保存するために、一時ディレクトリを取得
        temp_dir = tempfile.gettempdir()  # OSに依存せず、一時ファイル用のディレクトリを取得
        filename = f"{uuid.uuid4()}.{file.filename.split('.')[-1]}"
        file_path = os.path.join(temp_dir, filename)  # 一時ファイルの保存パスを組み立て

        # ファイルを一時保存
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # BlobStorageに画像をアップロード
        blob_client = container_client.get_blob_client(filename)
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        # アップロードしたBlobのURLを取得
        blob_url = blob_client.url

        # データベースに情報を保存
        store_image_metadata(filename, blob_url)

        # 一時ファイルを削除
        os.remove(file_path)

        return {
            "message": "画像が正常にアップロードされました", 
            "filename": filename,
            "blob_url": blob_url}

    except Exception as e:
        return {"error": str(e)}
    


# ▼▼▼以下、/category画面での処理▼▼▼  

# 画像のマスク処理関数
@app.post("/api/process")
async def process_mask(request: ProcessRequest):
    """
    ユーザーが描いたマスク領域を処理
    """
    print(f"===== マスク処理開始: image_id {request.image_id} =====")
    print(f"マスクデータ長さ: {len(request.mask_data) if request.mask_data else 'なし'}")
    
    try:
        # 元画像の情報を取得
        print("元画像情報取得中...")
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM images WHERE image_id = %s", (request.image_id,))
            image_info = cursor.fetchone()
            if not image_info:
                print(f"画像が見つかりません: {request.image_id}")
                raise HTTPException(status_code=404, detail="画像が見つかりません")
            print(f"元画像情報: {image_info}")
        finally:
            cursor.close()
            conn.close()

        # マスク処理
        mask_id = str(uuid.uuid4())
        mask_path = f"{UPLOAD_DIR}/masks/{mask_id}.png"
        print(f"生成されたmask_id: {mask_id}")
        print(f"マスク保存先: {mask_path}")
        
        # マスクデータを処理して保存
        print("マスク画像処理中...")
        if not process_mask_image(request.mask_data, mask_path):
            print("マスク処理失敗")
            raise HTTPException(status_code=500, detail="マスク処理に失敗しました")
        print("マスク処理成功")
        
        # データベースにマスク情報を保存
        print("マスク情報をデータベースに保存中...")
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO masks (mask_id, image_id, mask_path, created_at) VALUES (%s, %s, %s, NOW())",
                (mask_id, request.image_id, mask_path)
            )
            conn.commit()
            print("マスク情報保存成功")
        except Error as e:
            print(f"マスク保存エラー: {e}")
        finally:
            cursor.close()
            conn.close()
        
        print(f"===== マスク処理完了 =====")
        return {
            "success": True,
            "image_id": request.image_id,
            "mask_id": mask_id,
            "mask_path": mask_path,
            "public_url": f"/uploads/masks/{mask_id}.png"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"マスク処理エラー: {e}")
        raise HTTPException(status_code=500, detail=f"マスク処理エラー: {str(e)}")

# 画像処理ヘルパー関数
def process_mask_image(mask_data_base64, output_path):
    """
    Base64マスクデータを処理して保存
    """
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



# ▼▼▼以下、/material画面での処理▼▼▼
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



@app.post("/api/apply-material")
async def apply_material(request: MaterialRequest):
    """
    選択された素材をマスク領域に適用
    """
    print(f"===== 素材適用API開始 =====")
    print(f"リクエスト: image_id={request.image_id}, mask_id={request.mask_id}, material_id={request.material_id}")
    
    try:
        # 必要な情報を取得
        print("データベースから情報取得中...")
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            # 元画像情報
            cursor.execute("SELECT * FROM images WHERE image_id = %s", (request.image_id,))
            image_info = cursor.fetchone()
            if not image_info:
                print(f"画像が見つかりません: {request.image_id}")
                raise HTTPException(status_code=404, detail="画像が見つかりません")
            print(f"画像情報: {image_info}")
            
            # マスク情報
            cursor.execute("SELECT * FROM masks WHERE mask_id = %s", (request.mask_id,))
            mask_info = cursor.fetchone()
            if not mask_info:
                print(f"マスクが見つかりません: {request.mask_id}")
                raise HTTPException(status_code=404, detail="マスクが見つかりません")
            print(f"マスク情報: {mask_info}")
            
            # 素材情報
            cursor.execute("SELECT * FROM materials WHERE material_id = %s", (request.material_id,))
            material_info = cursor.fetchone()
            if not material_info:
                print(f"素材が見つかりません: {request.material_id}")
                raise HTTPException(status_code=404, detail="素材が見つかりません")
            print(f"素材情報: {material_info}")
        finally:
            cursor.close()
            conn.close()
        
        # 合成処理
        result_id = str(uuid.uuid4())
        result_path = f"{UPLOAD_DIR}/results/{result_id}.jpg"
        print(f"生成されたresult_id: {result_id}")
        print(f"結果保存先: {result_path}")
        
        # 素材を適用
        print("素材適用処理を実行中...")
        if not apply_material_to_image(
            image_info["original_path"], 
            mask_info["mask_path"], 
            material_info["material_path"], 
            result_path
        ):
            print("素材適用処理に失敗")
            raise HTTPException(status_code=500, detail="素材適用に失敗しました")
        
        # 結果をデータベースに保存
        print("結果をデータベースに保存中...")
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO results 
                (result_id, image_id, mask_id, material_id, result_path, created_at) 
                VALUES (%s, %s, %s, %s, %s, NOW())
                """,
                (result_id, request.image_id, request.mask_id, request.material_id, result_path)
            )
            conn.commit()
            print("結果保存成功")
        except Error as e:
            print(f"結果保存エラー: {e}")
        finally:
            cursor.close()
            conn.close()
        
        print(f"===== 素材適用API完了 =====")
        return {
            "success": True,
            "result_id": result_id,
            "result_path": result_path,
            "public_url": f"/uploads/results/{result_id}.jpg"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"素材適用APIエラー: {e}")
        raise HTTPException(status_code=500, detail=f"素材適用エラー: {str(e)}")
    


# 追加: ルートエンドポイント　よくわかんないけど必要らしいのでコメントアウト外した（4/4 羽田野）
@app.get("/")
async def root():
    return {"message": "FastAPI is running!"}

# Uvicornサーバー起動用　ローカル実行用に追加（4/4 羽田野）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

