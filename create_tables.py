import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

# .env ファイルから環境変数を読み込む
load_dotenv()

# データベース接続情報
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

# テーブル作成用SQL
create_tables_sql = [
    """
    -- マスク情報テーブル
    CREATE TABLE IF NOT EXISTS masks (
        mask_id VARCHAR(255) PRIMARY KEY,
        image_id VARCHAR(255) NOT NULL,  -- upload_imagesのfilenameを参照
        mask_path VARCHAR(255) NOT NULL,
        created_at DATETIME NOT NULL
    );
    """,
    """
    -- 素材テーブル（存在しない場合のみ）
    CREATE TABLE IF NOT EXISTS materials (
        material_id VARCHAR(255) PRIMARY KEY,
        category VARCHAR(255) NOT NULL,
        material_path VARCHAR(255) NOT NULL,
        material_name VARCHAR(255) NOT NULL,
        created_at DATETIME NOT NULL
    );
    """,
    """
    -- 結果テーブル
    CREATE TABLE IF NOT EXISTS results (
        result_id VARCHAR(255) PRIMARY KEY,
        image_id VARCHAR(255) NOT NULL,  -- upload_imagesのfilenameを参照
        mask_id VARCHAR(255) NOT NULL,
        material_id VARCHAR(255) NOT NULL,
        result_path VARCHAR(255) NOT NULL,
        created_at DATETIME NOT NULL
    );
    """
]

try:
    # データベースに接続
    connection = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    
    if connection.is_connected():
        cursor = connection.cursor()
        
        # 各テーブル作成クエリを実行
        for query in create_tables_sql:
            cursor.execute(query)
            print(f"テーブル作成クエリが実行されました")
        
        connection.commit()
        print("テーブルが正常に作成されました")
        
except Error as e:
    print(f"データベースエラー: {e}")

finally:
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("データベース接続を閉じました")