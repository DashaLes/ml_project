from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

class Item(BaseModel):
    query: str  #Приняли текст
    docs: list  #Приняли массив

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict/")
def predict(item: Item):

    model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

    query_emb = model.encode(item.query)    #Закодировали текст
    doc_emb = model.encode(item.docs)       #Закодировали массив

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()   #Применили модель

    doc_score_pairs = list(zip(item.docs, scores))  #Разобрали результат

    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)     #Отсортировали результат
    
    return {"result": doc_score_pairs}  #Вернули
    #Мы поняли весь код!
