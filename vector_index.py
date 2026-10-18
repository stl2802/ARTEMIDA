import json
import numpy as np

# Пути к файлам из задания 4 (TF-IDF + SVD)
DOC_VECTORS_SVD_FILE = "npy/doc_vectors_svd.npy"
DOC_IDS_FILE = "json/doc_ids.json"

class VectorIndex:
    def __init__(self):
        self.doc_vectors = None  # SVD-векторы документов
        self.doc_ids = None      # Соответствие ID документов
    
    def load_vectors(self):
        """
        Загрузка векторных представлений документов и их идентификаторов.
        Используются те же векторы, что и в задании 4 (TF-IDF + SVD).
        """
        self.doc_vectors = np.load(DOC_VECTORS_SVD_FILE)
        
        with open(DOC_IDS_FILE, 'r', encoding='utf-8') as f:
            self.doc_ids = json.load(f)
        
        print(f"Загружены векторные представления:")
        print(f"   Размерность матрицы: {self.doc_vectors.shape}")
        print(f"   Количество документов: {len(self.doc_ids)}")
        print(f"   Размерность векторов: {self.doc_vectors.shape[1]}")
        
        assert len(self.doc_ids) == self.doc_vectors.shape[0], \
            f"Несоответствие: {len(self.doc_ids)} ID, но {self.doc_vectors.shape[0]} векторов"
        print(f"Соответствие между векторами и ID документов проверено")
        
        return self.doc_vectors, self.doc_ids


def main():
    index = VectorIndex()
    vectors, doc_ids = index.load_vectors()
    
    print(f"\n Статистика векторного пространства:")
    print(f"   Тип данных: {vectors.dtype}")
    print(f"   Диапазон значений: [{vectors.min():.4f}, {vectors.max():.4f}]")
    print(f"   Среднее значение: {vectors.mean():.4f}")
    print(f"   Норма первого вектора: {np.linalg.norm(vectors[0]):.4f}")
    print(f"   Первый документ: {doc_ids[0]}")
    print(f"   Последний документ: {doc_ids[-1]}")


if __name__ == "__main__":
    main()