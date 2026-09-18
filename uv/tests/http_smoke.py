from urllib.request import urlopen

with urlopen("http://127.0.0.1:8507/_stcore/health", timeout=10) as response:
    body = response.read().decode("utf-8")
    assert response.status == 200
    assert body.strip() == "ok"

print("Streamlit HTTP health check passed")

