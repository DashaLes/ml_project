from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_execute_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
    
def test_read_query_predict():
    response = client.post("/predict/",
        json={
            "query": ["I had a similar conversation with her."],
            "docs": [
                "I had a similar conversation with him.",
                "I had a conversation."
            ]
        }
    )
    json_data = response.json()

    assert response.status_code == 422
    assert json_data['detail'][0]['msg'] == "str type expected"
    
def test_read_docs_predict():
    response = client.post("/predict/",
        json={
            "query": "I had a similar conversation with her.",
            "docs": [[
                "I had a similar conversation with him.",
                "I had a conversation."
            ]]
        }
    )
    json_data = response.json()

    assert response.status_code == 200
    assert json_data['error'] == "Список \"docs\" должен состоять из строк"
    
def test_execute_predict():
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
