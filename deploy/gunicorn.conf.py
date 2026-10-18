bind = "127.0.0.1:8000"
workers = 1          # CLIP + 中文模型常驻约 1.2 GB，单 worker
threads = 4
timeout = 180
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
