import sys
import argparse

def cmd_create_data(args):
    """создание необходимых документов"""
    from create_data.literature_installer import main
    main()

def cmd_corpus(args):
    """Обработка корпуса в json/corpus.json"""
    from create_data.process_corpus import main
    main()


def cmd_preprocess(args):
    """Предобработка корпуса"""
    from preprocess_corpus import main
    main()


def cmd_vocabulary(args):
    """Построение словаря"""
    from vectoryzing.vectorize_corpus import build_vocabulary
    build_vocabulary()


def cmd_lexical(args):
    """Лексический индекс (BM25)"""
    if args.mode == "build":
        from searches.lecsika_search import build_mode
        build_mode("json/preprocessed_corpus.json", "json/inverted_index.json")
    elif args.mode == "search":
        from searches.lecsika_search import search_mode
        search_mode("json/inverted_index.json", "json/queries.json")


def cmd_tfidf(args):
    """TF-IDF векторы"""
    from tfidf_search import TfidfSearcher
    searcher = TfidfSearcher()
    if args.mode == "build":
        searcher.build()
    elif args.mode == "svd":
        searcher.load()
        searcher.apply_svd(k=args.k)
    elif args.mode == "search":
        searcher.load()
        _run_search(searcher.search, searcher.forward_index)


def cmd_neural(args):
    """Нейронные векторы"""
    from searches.neural_search import NeuralSearcher
    searcher = NeuralSearcher()
    if args.mode == "build":
        searcher.build()
    elif args.mode == "search":
        searcher.load()
        _run_search(searcher.search, searcher.forward_index, get_all=lambda r: (r[0], r[1], r[2], r[3]))


def cmd_hnsw(args):
    """HNSW индекс"""
    if args.mode == "build":
        from searches.hnsw_search import build_index
        build_index()
    elif args.mode == "search":
        from searches.hnsw_search import HnswSearcher
        searcher = HnswSearcher()
        _run_search(searcher.approximate_search, searcher.forward_index, 
                    get_all=lambda r: (r[0], None, None, r[1]), score_name="similarity")


def cmd_hybrid(args):
    """Гибридный поиск"""
    from searches.hybrid_search import HybridSearcher
    if args.mode == "build":
        HybridSearcher()
        print("Все индексы загружены успешно!")
    elif args.mode == "search":
        searcher = HybridSearcher()
        _run_hybrid_search(searcher, alpha=args.alpha)


def cmd_smart_hybrid(args):
    """Умный гибридный поиск с автоматическим выбором метода"""
    from searches.smart_hybrid_search import search_mode
    search_mode()


def cmd_ltr(args):
    """Learning to Rank"""
    if args.mode == "train":
        from train_ltr import train_model
        train_model()
    elif args.mode == "prepare":
        from neural_train_process_data.prepare_ltr_dataset import prepare_dataset
        prepare_dataset()
    elif args.mode == "search":
        from searches.ltr_search import LTRSearcher
        searcher = LTRSearcher()
        _run_ltr_search(searcher)


def cmd_index_all(args):
    """Построение всех индексов подряд"""
    print("=" * 60)
    print("ПОЛНАЯ ИНДЕКСАЦИЯ")
    print("=" * 60)

    print("\n[1/5] Словарь...")
    from vectoryzing.vectorize_corpus import build_vocabulary
    build_vocabulary()

    print("\n[2/5] Лексический индекс...")
    from searches.lecsika_search import build_mode
    build_mode("json/preprocessed_corpus.json", "json/inverted_index.json")

    print("\n[3/5] TF-IDF + SVD...")
    from tfidf_search import TfidfSearcher
    tfidf = TfidfSearcher()
    tfidf.build()
    tfidf.apply_svd(k=args.k)

    print("\n[4/5] HNSW индекс...")
    from searches.hnsw_search import build_index
    build_index()

    print("\n[5/5] Нейронные векторы...")
    from searches.neural_search import NeuralSearcher
    neural = NeuralSearcher()
    neural.build()

    print("\n" + "=" * 60)
    print("ИНДЕКСАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)


def cmd_init(args):
    """Полная инициализация проекта с нуля - генерация всех файлов"""
    print("-" * 70)
    print("ПОЛНАЯ ИНИЦИАЛИЗАЦИЯ ПРОЕКТА")
    print("-" * 70)
    print("\nЭтот процесс выполнит все шаги для генерации всех файлов проекта:")
    print("  1. Обработка корпуса из literature/")
    print("  2. Предобработка корпуса")
    print("  3. Построение словаря")
    print("  4. Лексический индекс (BM25)")
    print("  5. TF-IDF векторы + SVD")
    print("  6. HNSW индекс")
    print("  7. Нейронные векторы (multilingual)")
    print("  8. Подготовка LTR датасета")
    print("  9. Разделение LTR датасета на train/test")
    print("  10. Обучение LTR модели (LightGBM)")
    print("=" * 70)

    import os
    flag = False
    if not os.path.exists("literature"):
        print("\n" + "-" * 70)
        print("[1/10] создание тестовых данных")
        print("-" * 70)
        from create_data.literature_installer import main
        main()
        flag = True

    try:
        print("\n" + "-" * 70)
        print("[2/10] Обработка корпуса из literature/")
        print("-" * 70)
        from create_data.process_corpus import main as process_corpus_main
        process_corpus_main()

        print("\n" + "-" * 70)
        print("[3/10] Предобработка корпуса")
        print("-" * 70)
        from preprocess_corpus import main as preprocess_main
        preprocess_main()

        print("\n" + "-" * 70)
        print("[4/10] Построение словаря")
        print("-" * 70)
        from vectoryzing.vectorize_corpus import build_vocabulary
        build_vocabulary()

        print("\n" + "-" * 70)
        print("[5/10] Лексический индекс (BM25)")
        print("-" * 70)
        from searches.lecsika_search import build_mode
        build_mode("json/preprocessed_corpus.json", "json/inverted_index.json")

        print("\n" + "-" * 70)
        print("[6/10] TF-IDF векторы + SVD")
        print("-" * 70)
        from tfidf_search import TfidfSearcher
        tfidf = TfidfSearcher()
        tfidf.build()
        tfidf.apply_svd(k=args.k)

        print("\n" + "=" * 70)
        print("[7/10] HNSW индекс")
        print("=" * 70)
        from searches.hnsw_search import build_index
        build_index()

        print("\n" + "-" * 70)
        print("[8/10] Нейронные векторы (multilingual)")
        print("-" * 70)
        from searches.neural_search import NeuralSearcher
        neural = NeuralSearcher()
        neural.build()

        print("\n" + "=" * 70)
        print("[9/10] Подготовка LTR датасета")
        print("=" * 70)
        from neural_train_process_data.prepare_ltr_dataset import prepare_dataset
        prepare_dataset()
        
    
        print("\n" + "-" * 70)
        print("[9.5/10] Разделение LTR датасета на train/test")
        print("-" * 70)
        from neural_train_process_data.split_dataset import split_dataset
        split_dataset()

        print("\n" + "-" * 70)
        print("[10/10] Обучение LTR модели (LightGBM)")
        print("-" * 70)
        from train_ltr import train_model
        train_model()

        print("\n" + "-" * 70)
        print("ПОЛНАЯ ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        print("-" * 70)
        print("\nВсе файлы сгенерированы:")
        if flag:
            print("  - literature/")
        print("  - json/corpus.json")
        print("  - json/preprocessed_corpus.json")
        print("  - json/inverted_index.json")
        print("  - json/vocabulary.json")
        print("  - json/doc_ids.json")
        print("  - json/neural_doc_ids.json")
        print("  - json/queries.json")
        print("  - json/ltr_dataset.json")
        print("  - json/ltr_train.json")
        print("  - json/ltr_test.json")
        print("  - npy/doc_vectors_svd.npy")
        print("  - npy/svd_vt.npy")
        print("  - npy/neural_doc_vectors.npy")
        print("  - bin/hnsw_index.bin")
        print("  - model/ltr_model.pkl")
        print("  - model/feature_order.json")
        print("\nТеперь можно использовать все команды поиска:")
        print("  python main.py smart-hybrid  - умный гибридный поиск")
        print("  python main.py hybrid search   - гибридный поиск")
        print("  python main.py neural search  - нейронный поиск")
        print("  python main.py ltr search     - LTR поиск")
        print("  python main.py hnsw search     - HNSW поиск")

    except Exception as e:
        print("\n" + "-" * 70)
        print("ОШИБКА ПРИ ИНИЦИАЛИЗАЦИИ")
        print("-" * 70)
        print(f"\n{e}")
        import traceback
        traceback.print_exc()


def cmd_benchmark_alpha(args):
    """Бенчмарк влияния параметра α"""
    from benchmarks.benchmark_alpha import run_benchmark
    run_benchmark()


def cmd_benchmark_hnsw(args):
    """Бенчмарк HNSW vs точный поиск"""
    from benchmarks.benchmark_hnsw import main
    main()


def cmd_benchmark_ltr(args):
    """Бенчмарк LTR vs гибридный поиск"""
    from benchmarks.benchmark_ltr import main
    main()


def _run_search(search_func, forward_index, get_all=None, score_name="score"):
    """Интерактивный поиск"""
    import json
    queries = _load_test_queries()
    
    while True:
        if queries:
            query = queries.pop(0)
            print(f"\nЗапрос: '{query}'")
        else:
            query = input("\nВведите запрос (или 'exit'): ")
            if query.lower() == 'exit':
                break
        
        results = search_func(query, top_k=5)
        if not results:
            print("Не найдено")
            continue
        
        print(f"\nТоп-5 результатов:")
        for rank, item in enumerate(results, 1):
            if get_all:
                doc_id, title, summary, score = get_all(item)
            else:
                doc_id, score = item
                title = forward_index[doc_id]["title"]
                summary = forward_index[doc_id]["summary"]
            
            print(f"{rank}. [{doc_id}] ({score_name}: {score:.4f})")
            print(f"   Title: {title}")
            if summary:
                print(f"   Summary: {summary[:100]}...")


def _run_hybrid_search(searcher, alpha=0.5):
    """Интерактивный гибридный поиск"""
    queries = _load_test_queries()
    
    while True:
        if queries:
            query = queries.pop(0)
            print(f"\n{'─' * 60}")
            print(f"Запрос: '{query}'")
            print(f"α = {alpha}")
        else:
            query = input("\nВведите запрос (или 'exit'): ")
            if query.lower() == 'exit':
                break
            try:
                a = input("α [0-1, Enter=0.5]: ").strip()
                alpha = float(a) if a else 0.5
            except:
                alpha = 0.5
        
        results = searcher.hybrid_search(query, top_k=5, alpha=alpha)
        if not results:
            print("Не найдено")
            continue
        
        print(f"\nТоп-5 (α={alpha:.2f}):")
        for rank, res in enumerate(results, 1):
            print(f"\n{rank}. [{res['doc_id']}] (hybrid: {res['hybrid_score']:.4f})")
            print(f"   Title: {res['title']}")
            print(f"   BM25: {res['bm25_score']:.4f} | Cosine: {res['cosine_score']:.4f}")


def _run_ltr_search(searcher):
    """Интерактивный LTR поиск"""
    while True:
        query = input("\nВведите запрос (или 'exit'): ")
        if query.lower() == 'exit':
            break
        
        results = searcher.search(query, top_k=5)
        if not results:
            print("Не найдено")
            continue
        
        print(f"\nРезультаты LTR:")
        for rank, res in enumerate(results, 1):
            entry = searcher.feature_extractor.forward_index[res["doc_id"]]
            print(f"\n{rank}. [{res['doc_id']}] (LTR: {res['ltr_score']:.4f})")
            print(f"   Title: {entry.title}")
            print(f"   Summary: {entry.summary[:100]}...")


def _load_test_queries():
    """Загрузка тестовых запросов"""
    import json
    try:
        with open("json/queries.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        return (
            [q["en"] if isinstance(q, dict) else q for q in data["queries"]["exact_terms"][:2]] +
            [q["en"] if isinstance(q, dict) else q for q in data["queries"]["detailed_queries"][:2]]
        )
    except:
        return []


def main():
    parser = argparse.ArgumentParser(
        description="ARTEMIDA - гибридная поисковая система",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main.py init                   - полная инициализация проекта с нуля
  python main.py index-all              - полная индексация
  python main.py tfidf build            - TF-IDF векторы
  python main.py hybrid search          - гибридный поиск
  python main.py hybrid search -a 0.7   - гибридный поиск с alpha=0.7
  python main.py smart-hybrid           - умный гибридный поиск (автовыбор: neural/lexical/hybrid/ltr/hnsw)
  python main.py ltr train              - обучение LTR модели
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Команды")
    
    p_create_data = subparsers.add_parser("create-data", help="Создание литературы для поиска в ней")
    p_create_data.set_defaults(func=cmd_create_data)
    
    p_corpus = subparsers.add_parser("corpus", help="Обработка корпуса из literature/")
    p_corpus.set_defaults(func=cmd_corpus)
    
    p_prep = subparsers.add_parser("preprocess", help="Предобработка корпуса")
    p_prep.set_defaults(func=cmd_preprocess)
    
    p_vocab = subparsers.add_parser("vocabulary", help="Построение словаря")
    p_vocab.set_defaults(func=cmd_vocabulary)
    
    p_lex = subparsers.add_parser("lexical", help="Лексический поиск (BM25)")
    p_lex.add_argument("mode", choices=["build", "search"], help="Режим")
    p_lex.set_defaults(func=cmd_lexical)
    
    p_tfidf = subparsers.add_parser("tfidf", help="TF-IDF векторы")
    p_tfidf.add_argument("mode", choices=["build", "svd", "search"], help="Режим")
    p_tfidf.add_argument("-k", type=int, default=100, help="Размерность SVD")
    p_tfidf.set_defaults(func=cmd_tfidf)
    
    p_neural = subparsers.add_parser("neural", help="Нейронные векторы")
    p_neural.add_argument("mode", choices=["build", "search"], help="Режим")
    p_neural.set_defaults(func=cmd_neural)
    
    p_hnsw = subparsers.add_parser("hnsw", help="HNSW индекс")
    p_hnsw.add_argument("mode", choices=["build", "search"], help="Режим")
    p_hnsw.set_defaults(func=cmd_hnsw)
    
    p_hybrid = subparsers.add_parser("hybrid", help="Гибридный поиск")
    p_hybrid.add_argument("mode", choices=["build", "search"], help="Режим")
    p_hybrid.add_argument("-a", "--alpha", type=float, default=0.5, help="Вес BM25 (0-1), alpha")
    p_hybrid.set_defaults(func=cmd_hybrid)

    p_smart = subparsers.add_parser("smart-hybrid", help="Умный гибридный поиск (автовыбор: neural/lexical/hybrid/ltr/hnsw)")
    p_smart.set_defaults(func=cmd_smart_hybrid)

    p_ltr = subparsers.add_parser("ltr", help="Learning to Rank")
    p_ltr.add_argument("mode", choices=["prepare", "train", "search"], help="Режим")
    p_ltr.set_defaults(func=cmd_ltr)
    
    p_all = subparsers.add_parser("index-all", help="Построение всех индексов")
    p_all.add_argument("-k", type=int, default=100, help="Размерность SVD")
    p_all.set_defaults(func=cmd_index_all)

    p_init = subparsers.add_parser("init", help="Полная инициализация проекта с нуля")
    p_init.add_argument("-k", type=int, default=100, help="Размерность SVD")
    p_init.set_defaults(func=cmd_init)

    p_bench = subparsers.add_parser("benchmark", help="Бенчмарки")
    p_bench.add_argument("type", choices=["alpha", "hnsw", "ltr"], 
                         help="alpha - влияние α, hnsw - сравнение с точным, ltr - сравнение с гибридным")
    p_bench.set_defaults(func=lambda a: {
        "alpha": cmd_benchmark_alpha,
        "hnsw": cmd_benchmark_hnsw,
        "ltr": cmd_benchmark_ltr
    }[a.type](a))
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
