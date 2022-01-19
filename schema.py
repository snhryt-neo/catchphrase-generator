SCHEMA = [
    {
        "description": "submit日時",
        "name": "created_at_jst",
        "type": "DATETIME",
        "mode": "REQUIRED",
    },
    {
        "description": "入力キャッチコピー",
        "name": "input_phrase",
        "type": "STRING",
        "mode": "REQUIRED",
    },
    {
        "description": "最終的に決定したキャッチコピー",
        "name": "decided_phrase",
        "type": "STRING",
        "mode": "NULLABLE",
    },
    {
        "description": "生成されたキャッチコピー",
        "name": "generated_phrase",
        "type": "STRING",
        "mode": "REQUIRED",
    },
    {
        "description": "生成されたキャッチコピーに対するフィードバック",
        "name": "is_good",
        "type": "BOOLEAN",
        "mode": "REQUIRED",
    },
]
