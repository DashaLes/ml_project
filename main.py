from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

class Item(BaseModel):
    query: str  #Приняли текст
    docs: list  #Приняли список

app = FastAPI()


@app.get("/")
def root():
    """Возвращает \"Hello World\""""
    return {"message": "Hello World"}

@app.post("/predict/")
def predict(item: Item):
    """\"query\" принимает строку. \"docs\" принимает список строк для анализа сходства с \"query\". Функция возвращает список, в котором хранятся строки и результаты их анализов."""
    
    for elem in item.docs:
        if (isinstance(elem, str) != True):
            return {"error": "Список \"docs\" должен состоять из строк"}

    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

    query_emb = model.encode(item.query)    #Закодировали текст
    doc_emb = model.encode(item.docs)       #Закодировали список

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()   #Применили модель

    doc_score_pairs = list(zip(item.docs, scores))  #Разобрали результат

    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)     #Отсортировали результат
    
    return {"result": doc_score_pairs}  #Вернули
    #Мы поняли весь код!
