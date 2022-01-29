from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
    
def test_read_predict():
    response = client.post("/predict/",
        json={
            "query": "I had a similar conversation with her.",
            "docs": [
                "I had a similar conversation with him.",
                "I had a conversation."
            ]
        }
    )
    json_data = response.json()

    assert response.status_code == 200
    assert json_data['result'][0][0] == "I had a similar conversation with him."
    assert json_data['result'][1][0] == "I had a conversation."
