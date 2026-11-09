---
title: "119 вопросов по LLM и RAG — перевод и разметка"
language: ru
original_language: en
document_type: question_bank
item_count: 119
section_count: 16
answers_present: false
chunking_strategy: one_question_per_chunk
---

# 119 вопросов и кейсов по LLM, RAG и смежным технологиям

Полный перевод исходного списка. Каждый вопрос снабжён ID и тегами для RAG-индексации. Список содержит **вопросы и темы кейсов, но не ответы**.

## Оглавление

1. [Промпт-инжиниринг и основы LLM](#prompt_engineering)
2. [RAG — генерация с дополнением найденным контекстом](#rag)
3. [Оцифровка документов и чанкинг](#document_chunking)
4. [Эмбеддинги и модели векторных представлений](#embeddings)
5. [Внутреннее устройство векторных баз данных](#vector_databases)
6. [Продвинутые алгоритмы поиска](#advanced_retrieval)
7. [Внутреннее устройство языковых моделей](#transformer_internals)
8. [Контролируемое дообучение LLM (SFT)](#supervised_fine_tuning)
9. [Выравнивание предпочтений: RLHF и DPO](#preference_alignment)
10. [Оценка качества LLM-систем](#llm_evaluation)
11. [Методы контроля галлюцинаций](#hallucination_control)
12. [Развёртывание и оптимизация LLM](#deployment)
13. [Агентные системы](#agents)
14. [Взлом промптов и защита от атак](#prompt_security)
15. [Разные вопросы об LLM и RAG](#miscellaneous)
16. [Практические кейсы](#case_studies)

## Рекомендации по индексации

- Один вопрос — один независимый чанк. Не смешивайте вопросы из разных тем в один фрагмент.
- В `text_for_embedding` включены раздел, русская формулировка вопроса и теги; это поле готово для векторизации.
- Используйте поля `id`, `section_id`, `tags`, `source_line` для фильтрации и обратных ссылок.
- Поле `question_en` в JSONL сохраняет исходный английский текст для мультиязычного поиска.
- `has_answer: false` означает, что исходник не содержит проверенного ответа. При генерации ответа требуется другой источник знаний.

<a id="prompt_engineering"></a>
## 01. Промпт-инжиниринг и основы LLM

### RAG-Q001 — В чём разница между предсказательным (дискриминативным) и генеративным искусственным интеллектом?

**Теги:** `prompt_engineering`, `llm`, `prompting`, `discriminative_ai`, `generative_ai`
**Источник:** строка 22 исходного файла.

### RAG-Q002 — Что такое большая языковая модель (LLM) и как обучают такие модели?

**Теги:** `prompt_engineering`, `llm`, `prompting`, `llm_training`
**Источник:** строка 23 исходного файла.

### RAG-Q003 — Что такое токен в языковой модели?

**Теги:** `prompt_engineering`, `llm`, `prompting`, `tokenization`
**Источник:** строка 24 исходного файла.

### RAG-Q004 — Как оценить стоимость эксплуатации облачных (SaaS) и открытых (open-source) LLM?

**Теги:** `prompt_engineering`, `llm`, `prompting`, `cost_estimation`
**Источник:** строка 25 исходного файла.

### RAG-Q005 — Что означает параметр temperature (температура) и как его настраивать?

**Теги:** `prompt_engineering`, `llm`, `prompting`, `temperature`
**Источник:** строка 26 исходного файла.

### RAG-Q006 — Какие существуют стратегии декодирования для выбора следующего токена?

**Теги:** `prompt_engineering`, `llm`, `prompting`, `decoding`
**Источник:** строка 27 исходного файла.

### RAG-Q007 — Какими способами можно задавать условия остановки генерации в LLM?

**Теги:** `prompt_engineering`, `llm`, `prompting`, `stopping_criteria`
**Источник:** строка 28 исходного файла.

### RAG-Q008 — Как использовать стоп-последовательности (stop sequences) в LLM?

**Теги:** `prompt_engineering`, `llm`, `prompting`, `stop_sequences`
**Источник:** строка 29 исходного файла.

### RAG-Q009 — Как устроена базовая структура промпта (запроса к модели)?

**Теги:** `prompt_engineering`, `llm`, `prompting`, `prompt_structure`
**Источник:** строка 30 исходного файла.

### RAG-Q010 — Что такое обучение в контексте (in-context learning)?

**Теги:** `prompt_engineering`, `llm`, `prompting`, `in_context_learning`
**Источник:** строка 31 исходного файла.

### RAG-Q011 — Какие существуют виды промпт-инжиниринга?

**Теги:** `prompt_engineering`, `llm`, `prompting`, `prompt_types`
**Источник:** строка 32 исходного файла.

### RAG-Q012 — Что нужно учитывать при составлении промптов с несколькими примерами (few-shot prompting)?

**Теги:** `prompt_engineering`, `llm`, `prompting`, `few_shot`
**Источник:** строка 33 исходного файла.

### RAG-Q013 — Какие стратегии помогают писать качественные промпты?

**Теги:** `prompt_engineering`, `llm`, `prompting`, `prompt_best_practices`
**Источник:** строка 34 исходного файла.

### RAG-Q014 — Что такое галлюцинации LLM и как снижать их вероятность с помощью промпт-инжиниринга?

**Теги:** `prompt_engineering`, `llm`, `prompting`, `hallucination_mitigation`
**Источник:** строка 35 исходного файла.

### RAG-Q015 — Как улучшить способность LLM к рассуждению с помощью промпт-инжиниринга?

**Теги:** `prompt_engineering`, `llm`, `prompting`, `reasoning`
**Источник:** строка 36 исходного файла.

### RAG-Q016 — Как улучшить рассуждения LLM, если промпт с цепочкой рассуждений (Chain-of-Thought, CoT) не сработал?

**Теги:** `prompt_engineering`, `llm`, `prompting`, `chain_of_thought`
**Источник:** строка 37 исходного файла.

<a id="rag"></a>
## 02. RAG — генерация с дополнением найденным контекстом

### RAG-Q017 — Как повысить точность, надёжность и проверяемость ответов LLM?

**Теги:** `rag`, `retrieval_augmented_generation`, `answer_quality`
**Источник:** строка 45 исходного файла.

### RAG-Q018 — Как работает RAG (Retrieval-Augmented Generation)?

**Теги:** `rag`, `retrieval_augmented_generation`, `rag_pipeline`
**Источник:** строка 46 исходного файла.

### RAG-Q019 — Какие преимущества даёт использование RAG?

**Теги:** `rag`, `retrieval_augmented_generation`, `rag_benefits`
**Источник:** строка 47 исходного файла.

### RAG-Q020 — В каких случаях следует выбирать дообучение (fine-tuning) вместо RAG?

**Теги:** `rag`, `retrieval_augmented_generation`, `rag_vs_fine_tuning`
**Источник:** строка 48 исходного файла.

### RAG-Q021 — Какие архитектурные подходы используют для адаптации LLM к закрытым или корпоративным данным?

**Теги:** `rag`, `retrieval_augmented_generation`, `private_data_architecture`
**Источник:** строка 49 исходного файла.

<a id="document_chunking"></a>
## 03. Оцифровка документов и чанкинг

### RAG-Q022 — Что такое чанкинг (chunking) и зачем разбивать данные на фрагменты?

**Теги:** `document_chunking`, `document_processing`, `chunking`, `chunking_definition`
**Источник:** строка 57 исходного файла.

### RAG-Q023 — Какие факторы влияют на размер чанка?

**Теги:** `document_chunking`, `document_processing`, `chunking`, `chunk_size`
**Источник:** строка 58 исходного файла.

### RAG-Q024 — Какие существуют методы чанкинга?

**Теги:** `document_chunking`, `document_processing`, `chunking`, `chunking_methods`
**Источник:** строка 59 исходного файла.

### RAG-Q025 — Как подобрать оптимальный размер чанка?

**Теги:** `document_chunking`, `document_processing`, `chunking`, `chunk_size_optimization`
**Источник:** строка 60 исходного файла.

### RAG-Q026 — Как лучше оцифровывать и разбивать на чанки сложные документы, например годовые отчёты?

**Теги:** `document_chunking`, `document_processing`, `chunking`, `document_digitization`
**Источник:** строка 61 исходного файла.

### RAG-Q027 — Как обрабатывать таблицы при чанкинге?

**Теги:** `document_chunking`, `document_processing`, `chunking`, `table_chunking`
**Источник:** строка 62 исходного файла.

### RAG-Q028 — Как обрабатывать очень большие таблицы, чтобы улучшить поиск?

**Теги:** `document_chunking`, `document_processing`, `chunking`, `large_tables`
**Источник:** строка 63 исходного файла.

### RAG-Q029 — Как обрабатывать элементы списков при чанкинге?

**Теги:** `document_chunking`, `document_processing`, `chunking`, `list_chunking`
**Источник:** строка 64 исходного файла.

### RAG-Q030 — Как построить промышленный конвейер обработки документов и их индексации?

**Теги:** `document_chunking`, `document_processing`, `chunking`, `ingestion_pipeline`
**Источник:** строка 65 исходного файла.

### RAG-Q031 — Как обрабатывать графики и диаграммы в RAG?

**Теги:** `document_chunking`, `document_processing`, `chunking`, `chart_understanding`
**Источник:** строка 66 исходного файла.

<a id="embeddings"></a>
## 04. Эмбеддинги и модели векторных представлений

### RAG-Q032 — Что такое векторные представления (эмбеддинги) и модель эмбеддингов?

**Теги:** `embeddings`, `embedding_models`, `embedding_definition`
**Источник:** строка 74 исходного файла.

### RAG-Q033 — Как модель эмбеддингов применяется в приложениях на базе LLM?

**Теги:** `embeddings`, `embedding_models`, `embedding_usage`
**Источник:** строка 75 исходного файла.

### RAG-Q034 — Чем отличается получение эмбеддингов для короткого и длинного контента?

**Теги:** `embeddings`, `embedding_models`, `long_text_embeddings`
**Источник:** строка 76 исходного файла.

### RAG-Q035 — Как сравнивать качество моделей эмбеддингов на собственных данных?

**Теги:** `embeddings`, `embedding_models`, `embedding_benchmark`
**Источник:** строка 77 исходного файла.

### RAG-Q036 — Предположим, вы используете открытую модель эмбеддингов, но тесты показывают низкую точность поиска. Как улучшить качество этой модели и поиска?

**Теги:** `embeddings`, `embedding_models`, `embedding_optimization`
**Источник:** строка 78 исходного файла.

### RAG-Q037 — Какие шаги нужно предпринять, чтобы улучшить модель Sentence Transformers, используемую для эмбеддингов?

**Теги:** `embeddings`, `embedding_models`, `sentence_transformers`
**Источник:** строка 79 исходного файла.

<a id="vector_databases"></a>
## 05. Внутреннее устройство векторных баз данных

### RAG-Q038 — Что такое векторная база данных?

**Теги:** `vector_databases`, `vector_db`, `vector_index`, `vector_database_definition`
**Источник:** строка 87 исходного файла.

### RAG-Q039 — Чем векторная база данных отличается от традиционной базы данных?

**Теги:** `vector_databases`, `vector_db`, `vector_index`, `vector_vs_relational`
**Источник:** строка 88 исходного файла.

### RAG-Q040 — Как работает векторная база данных?

**Теги:** `vector_databases`, `vector_db`, `vector_index`, `vector_database_architecture`
**Источник:** строка 89 исходного файла.

### RAG-Q041 — Чем различаются векторный индекс, векторная СУБД и расширения для векторного поиска?

**Теги:** `vector_databases`, `vector_db`, `vector_index`, `vector_indexes`
**Источник:** строка 90 исходного файла.

### RAG-Q042 — Есть небольшой набор отзывов клиентов. Нужно находить наиболее похожие отзывы с максимально точным результатом; скорость поиска несущественна. Какую стратегию поиска выбрать и почему?

**Теги:** `vector_databases`, `vector_db`, `vector_index`, `exact_nearest_neighbor`
**Источник:** строка 91 исходного файла.

### RAG-Q043 — Как работают стратегии векторного поиска, например кластеризация и локально-чувствительное хеширование (LSH)?

**Теги:** `vector_databases`, `vector_db`, `vector_index`, `clustering_lsh`
**Источник:** строка 92 исходного файла.

### RAG-Q044 — Как кластеризация сокращает пространство поиска? Когда она даёт сбои и как смягчить эти проблемы?

**Теги:** `vector_databases`, `vector_db`, `vector_index`, `clustering_failure`
**Источник:** строка 93 исходного файла.

### RAG-Q045 — Что такое индекс на основе случайных проекций (random projection index)?

**Теги:** `vector_databases`, `vector_db`, `vector_index`, `random_projection`
**Источник:** строка 94 исходного файла.

### RAG-Q046 — Как работает индексирование методом локально-чувствительного хеширования (LSH; в оригинале указано LHS)?

**Теги:** `vector_databases`, `vector_db`, `vector_index`, `lsh_index`
**Источник:** строка 95 исходного файла.

### RAG-Q047 — Как работает индексирование с произведённым квантованием (product quantization, PQ)?

**Теги:** `vector_databases`, `vector_db`, `vector_index`, `product_quantization`
**Источник:** строка 96 исходного файла.

### RAG-Q048 — Сравните разные векторные индексы. Какой индекс вы выберете для конкретного проекта и почему?

**Теги:** `vector_databases`, `vector_db`, `vector_index`, `vector_index_selection`
**Источник:** строка 97 исходного файла.

### RAG-Q049 — Как выбрать подходящую метрику векторного сходства для конкретной задачи?

**Теги:** `vector_databases`, `vector_db`, `vector_index`, `similarity_metrics`
**Источник:** строка 98 исходного файла.

### RAG-Q050 — Какие типы фильтрации есть в векторных БД и с какими сложностями они связаны?

**Теги:** `vector_databases`, `vector_db`, `vector_index`, `metadata_filtering`
**Источник:** строка 99 исходного файла.

### RAG-Q051 — Как выбрать наиболее подходящую векторную базу данных под свои требования?

**Теги:** `vector_databases`, `vector_db`, `vector_index`, `vector_database_selection`
**Источник:** строка 100 исходного файла.

<a id="advanced_retrieval"></a>
## 06. Продвинутые алгоритмы поиска

### RAG-Q052 — Какие архитектурные подходы применяют для информационного и семантического поиска?

**Теги:** `advanced_retrieval`, `information_retrieval`, `search`, `retrieval_architecture`
**Источник:** строка 108 исходного файла.

### RAG-Q053 — Почему качество поиска настолько важно?

**Теги:** `advanced_retrieval`, `information_retrieval`, `search`, `search_quality`
**Источник:** строка 109 исходного файла.

### RAG-Q054 — Как добиться эффективного и точного поиска в больших наборах данных?

**Теги:** `advanced_retrieval`, `information_retrieval`, `search`, `large_scale_retrieval`
**Источник:** строка 110 исходного файла.

### RAG-Q055 — У клиента уже есть RAG-система, но она даёт неточные ответы. Вы выяснили, что причина — низкое качество retrieval. Какие шаги предпримете для его улучшения?

**Теги:** `advanced_retrieval`, `information_retrieval`, `search`, `retrieval_debugging`
**Источник:** строка 111 исходного файла.

### RAG-Q056 — Как работает поиск на основе ключевых слов (keyword-based retrieval)?

**Теги:** `advanced_retrieval`, `information_retrieval`, `search`, `keyword_search`
**Источник:** строка 112 исходного файла.

### RAG-Q057 — Как дообучать модели повторного ранжирования (reranking)?

**Теги:** `advanced_retrieval`, `information_retrieval`, `search`, `reranker_fine_tuning`
**Источник:** строка 113 исходного файла.

### RAG-Q058 — Какая метрика информационного поиска используется чаще всего и в каких ситуациях она не подходит?

**Теги:** `advanced_retrieval`, `information_retrieval`, `search`, `ir_metrics`
**Источник:** строка 114 исходного файла.

### RAG-Q059 — Вы разрабатываете систему вопросов и ответов наподобие Quora, где пользователи должны быстро находить наиболее релевантные ответы. Какую метрику вы выберете для оценки качества системы и почему?

**Теги:** `advanced_retrieval`, `information_retrieval`, `search`, `qa_ranking_metrics`
**Источник:** строка 115 исходного файла.

### RAG-Q060 — Какую метрику следует использовать для оценки рекомендательной системы?

**Теги:** `advanced_retrieval`, `information_retrieval`, `search`, `recommender_metrics`
**Источник:** строка 116 исходного файла.

### RAG-Q061 — Сравните метрики информационного поиска: когда и какую следует применять?

**Теги:** `advanced_retrieval`, `information_retrieval`, `search`, `metric_selection`
**Источник:** строка 117 исходного файла.

### RAG-Q062 — Как работает гибридный поиск (hybrid search)?

**Теги:** `advanced_retrieval`, `information_retrieval`, `search`, `hybrid_search`
**Источник:** строка 118 исходного файла.

### RAG-Q063 — Как объединить выдачи нескольких поисковых методов и согласовать их ранжирование в единый список?

**Теги:** `advanced_retrieval`, `information_retrieval`, `search`, `rank_fusion`
**Источник:** строка 119 исходного файла.

### RAG-Q064 — Как обрабатывать многошаговые (multi-hop) и многогранные (multifaceted) запросы?

**Теги:** `advanced_retrieval`, `information_retrieval`, `search`, `multi_hop_retrieval`
**Источник:** строка 120 исходного файла.

### RAG-Q065 — Какие методы позволяют улучшить качество поиска (retrieval)?

**Теги:** `advanced_retrieval`, `information_retrieval`, `search`, `retrieval_optimization`
**Источник:** строка 121 исходного файла.

<a id="transformer_internals"></a>
## 07. Внутреннее устройство языковых моделей

### RAG-Q066 — Подробно объясните механизм самовнимания (self-attention).

**Теги:** `transformer_internals`, `llm_architecture`, `transformers`, `self_attention`
**Источник:** строка 130 исходного файла.

### RAG-Q067 — Какие недостатки есть у механизма self-attention и как их преодолевать?

**Теги:** `transformer_internals`, `llm_architecture`, `transformers`, `attention_limitations`
**Источник:** строка 131 исходного файла.

### RAG-Q068 — Что такое позиционное кодирование (positional encoding)?

**Теги:** `transformer_internals`, `llm_architecture`, `transformers`, `positional_encoding`
**Источник:** строка 132 исходного файла.

### RAG-Q069 — Подробно объясните архитектуру Transformer.

**Теги:** `transformer_internals`, `llm_architecture`, `transformers`, `transformer_architecture`
**Источник:** строка 133 исходного файла.

### RAG-Q070 — Какие преимущества у Transformer по сравнению с LSTM?

**Теги:** `transformer_internals`, `llm_architecture`, `transformers`, `transformer_vs_lstm`
**Источник:** строка 134 исходного файла.

### RAG-Q071 — Чем локальное внимание (local attention) отличается от глобального (global attention)?

**Теги:** `transformer_internals`, `llm_architecture`, `transformers`, `local_vs_global_attention`
**Источник:** строка 135 исходного файла.

### RAG-Q072 — Почему Transformer требует много вычислительных ресурсов и памяти и как снизить эти затраты?

**Теги:** `transformer_internals`, `llm_architecture`, `transformers`, `memory_compute`
**Источник:** строка 136 исходного файла.

### RAG-Q073 — Как увеличить длину контекстного окна LLM?

**Теги:** `transformer_internals`, `llm_architecture`, `transformers`, `context_extension`
**Источник:** строка 137 исходного файла.

### RAG-Q074 — Как оптимизировать архитектуру Transformer, если словарь содержит 100 тысяч слов или токенов?

**Теги:** `transformer_internals`, `llm_architecture`, `transformers`, `vocabulary_scaling`
**Источник:** строка 138 исходного файла.

### RAG-Q075 — Большой словарь повышает вычислительные затраты, а маленький увеличивает проблему неизвестных токенов (OOV). Как найти оптимальный баланс?

**Теги:** `transformer_internals`, `llm_architecture`, `transformers`, `out_of_vocabulary`
**Источник:** строка 139 исходного файла.

### RAG-Q076 — Какие существуют архитектуры LLM и для каких задач подходит каждая?

**Теги:** `transformer_internals`, `llm_architecture`, `transformers`, `model_architectures`
**Источник:** строка 140 исходного файла.

<a id="supervised_fine_tuning"></a>
## 08. Контролируемое дообучение LLM (SFT)

### RAG-Q077 — Что такое дообучение (fine-tuning) и зачем оно нужно?

**Теги:** `supervised_fine_tuning`, `fine_tuning`, `sft`, `fine_tuning_definition`
**Источник:** строка 149 исходного файла.

### RAG-Q078 — В каких сценариях необходимо дообучать LLM?

**Теги:** `supervised_fine_tuning`, `fine_tuning`, `sft`, `fine_tuning_use_cases`
**Источник:** строка 150 исходного файла.

### RAG-Q079 — Как определить, стоит ли применять дообучение?

**Теги:** `supervised_fine_tuning`, `fine_tuning`, `sft`, `fine_tuning_decision`
**Источник:** строка 151 исходного файла.

### RAG-Q080 — Как обучить или настроить модель так, чтобы она отвечала только при наличии достаточного контекста?

**Теги:** `supervised_fine_tuning`, `fine_tuning`, `sft`, `abstention`
**Источник:** строка 152 исходного файла.

### RAG-Q081 — Как создавать датасеты для дообучения на задачах вопросов и ответов (Q&A)?

**Теги:** `supervised_fine_tuning`, `fine_tuning`, `sft`, `qa_training_data`
**Источник:** строка 153 исходного файла.

### RAG-Q082 — Как выбирать гиперпараметры для дообучения?

**Теги:** `supervised_fine_tuning`, `fine_tuning`, `sft`, `hyperparameters`
**Источник:** строка 154 исходного файла.

### RAG-Q083 — Как оценить требования к инфраструктуре для дообучения LLM?

**Теги:** `supervised_fine_tuning`, `fine_tuning`, `sft`, `training_infrastructure`
**Источник:** строка 155 исходного файла.

### RAG-Q084 — Как дообучать LLM на потребительском оборудовании?

**Теги:** `supervised_fine_tuning`, `fine_tuning`, `sft`, `consumer_gpu`
**Источник:** строка 156 исходного файла.

### RAG-Q085 — Какие существуют категории методов параметроэффективного дообучения (PEFT)?

**Теги:** `supervised_fine_tuning`, `fine_tuning`, `sft`, `peft`
**Источник:** строка 157 исходного файла.

### RAG-Q086 — Что такое катастрофическое забывание (catastrophic forgetting) в LLM?

**Теги:** `supervised_fine_tuning`, `fine_tuning`, `sft`, `catastrophic_forgetting`
**Источник:** строка 158 исходного файла.

### RAG-Q087 — Какие существуют методы дообучения на основе репараметризации?

**Теги:** `supervised_fine_tuning`, `fine_tuning`, `sft`, `reparameterization`
**Источник:** строка 159 исходного файла.

<a id="preference_alignment"></a>
## 09. Выравнивание предпочтений: RLHF и DPO

### RAG-Q088 — На каком этапе и при каких условиях следует выбирать выравнивание по предпочтениям (preference alignment) вместо SFT?

**Теги:** `preference_alignment`, `alignment`, `rlhf`, `dpo`, `sft_vs_alignment`
**Источник:** строка 167 исходного файла.

### RAG-Q089 — Что такое RLHF и как его применяют?

**Теги:** `preference_alignment`, `alignment`, `rlhf`, `dpo`
**Источник:** строка 168 исходного файла.

### RAG-Q090 — Что такое эксплуатация ошибок функции награды (reward hacking) в RLHF?

**Теги:** `preference_alignment`, `alignment`, `rlhf`, `dpo`, `reward_hacking`
**Источник:** строка 169 исходного файла.

### RAG-Q091 — Какие существуют методы выравнивания моделей по предпочтениям?

**Теги:** `preference_alignment`, `alignment`, `rlhf`, `dpo`, `alignment_methods`
**Источник:** строка 170 исходного файла.

<a id="llm_evaluation"></a>
## 10. Оценка качества LLM-систем

### RAG-Q092 — Как выбрать лучшую LLM для конкретного сценария применения на основе оценки качества?

**Теги:** `llm_evaluation`, `evaluation`, `llm_metrics`, `model_selection`
**Источник:** строка 178 исходного файла.

### RAG-Q093 — Как оценивать системы на основе RAG?

**Теги:** `llm_evaluation`, `evaluation`, `llm_metrics`, `rag_evaluation`
**Источник:** строка 179 исходного файла.

### RAG-Q094 — Какие метрики используют для оценки LLM?

**Теги:** `llm_evaluation`, `evaluation`, `llm_metrics`, `llm_evaluation_metrics`
**Источник:** строка 180 исходного файла.

### RAG-Q095 — Что такое цепочка проверки (Chain of Verification) и как она работает?

**Теги:** `llm_evaluation`, `evaluation`, `llm_metrics`, `chain_of_verification`
**Источник:** строка 181 исходного файла.

<a id="hallucination_control"></a>
## 11. Методы контроля галлюцинаций

### RAG-Q096 — Какие существуют виды галлюцинаций LLM?

**Теги:** `hallucination_control`, `hallucinations`, `grounding`, `hallucination_types`
**Источник:** строка 189 исходного файла.

### RAG-Q097 — Как контролировать галлюцинации на разных уровнях системы?

**Теги:** `hallucination_control`, `hallucinations`, `grounding`, `hallucination_prevention`
**Источник:** строка 190 исходного файла.

<a id="deployment"></a>
## 12. Развёртывание и оптимизация LLM

### RAG-Q098 — Почему квантование не снижает точность LLM?

**Теги:** `deployment`, `inference_optimization`, `quantization`
**Источник:** строка 198 исходного файла.

### RAG-Q099 — Какие способы оптимизации инференса LLM повышают пропускную способность (throughput)?

**Теги:** `deployment`, `inference_optimization`, `throughput`
**Источник:** строка 199 исходного файла.

### RAG-Q100 — Как уменьшить время ответа модели без приближённых механизмов внимания, например grouped-query attention?

**Теги:** `deployment`, `inference_optimization`, `latency`
**Источник:** строка 200 исходного файла.

<a id="agents"></a>
## 13. Агентные системы

### RAG-Q101 — Объясните базовые принципы работы AI-агентов и стратегии их реализации.

**Теги:** `agents`, `ai_agents`, `tool_use`, `agent_architecture`
**Источник:** строка 208 исходного файла.

### RAG-Q102 — Зачем нужны агенты и какие распространённые стратегии применяют при их реализации?

**Теги:** `agents`, `ai_agents`, `tool_use`, `agent_strategies`
**Источник:** строка 209 исходного файла.

### RAG-Q103 — Объясните подход ReAct с примером кода и перечислите его преимущества.

**Теги:** `agents`, `ai_agents`, `tool_use`, `react`
**Источник:** строка 210 исходного файла.

### RAG-Q104 — Объясните стратегию Plan-and-Execute для агентов.

**Теги:** `agents`, `ai_agents`, `tool_use`, `plan_and_execute`
**Источник:** строка 211 исходного файла.

### RAG-Q105 — Объясните использование OpenAI function calling (вызова инструментов) с примерами кода.

**Теги:** `agents`, `ai_agents`, `tool_use`, `function_calling`
**Источник:** строка 212 исходного файла.

### RAG-Q106 — Чем отличается OpenAI function calling от агентов LangChain?

**Теги:** `agents`, `ai_agents`, `tool_use`, `langchain_agents`
**Источник:** строка 213 исходного файла.

<a id="prompt_security"></a>
## 14. Взлом промптов и защита от атак

### RAG-Q107 — Что такое взлом промптов (prompt hacking) и почему он опасен?

**Теги:** `prompt_security`, `prompt_injection`, `security`, `prompt_hacking`
**Источник:** строка 221 исходного файла.

### RAG-Q108 — Какие существуют виды атак на промпты?

**Теги:** `prompt_security`, `prompt_injection`, `security`, `attack_types`
**Источник:** строка 222 исходного файла.

### RAG-Q109 — Какие методы защиты помогают против атак на промпты?

**Теги:** `prompt_security`, `prompt_injection`, `security`, `prompt_injection_defenses`
**Источник:** строка 223 исходного файла.

<a id="miscellaneous"></a>
## 15. Разные вопросы об LLM и RAG

### RAG-Q110 — Как оптимизировать общую стоимость LLM-системы?

**Теги:** `miscellaneous`, `llm`, `rag`, `optimization`, `llm_cost_optimization`
**Источник:** строка 231 исходного файла.

### RAG-Q111 — Что такое модели «смесь экспертов» (Mixture of Experts, MoE)?

**Теги:** `miscellaneous`, `llm`, `rag`, `optimization`, `mixture_of_experts`
**Источник:** строка 232 исходного файла.

### RAG-Q112 — Как построить RAG-систему промышленного уровня? Подробно опишите каждый компонент.

**Теги:** `miscellaneous`, `llm`, `rag`, `optimization`, `production_rag`
**Источник:** строка 233 исходного файла.

### RAG-Q113 — Что такое числовой формат FP8 и каковы его преимущества?

**Теги:** `miscellaneous`, `llm`, `rag`, `optimization`, `fp8`
**Источник:** строка 234 исходного файла.

### RAG-Q114 — Как обучать LLM с пониженной точностью вычислений без ухудшения качества?

**Теги:** `miscellaneous`, `llm`, `rag`, `optimization`, `low_precision_training`
**Источник:** строка 235 исходного файла.

### RAG-Q115 — Как рассчитать объём памяти для KV-кеша?

**Теги:** `miscellaneous`, `llm`, `rag`, `optimization`, `kv_cache`
**Источник:** строка 236 исходного файла.

### RAG-Q116 — Каковы размерности тензоров на каждом этапе блока многоголового внимания (multi-head attention)?

**Теги:** `miscellaneous`, `llm`, `rag`, `optimization`, `multi_head_attention_dimensions`
**Источник:** строка 237 исходного файла.

### RAG-Q117 — Как добиться того, чтобы слой внимания учитывал нужные части входных данных?

**Теги:** `miscellaneous`, `llm`, `rag`, `optimization`, `attention_focus`
**Источник:** строка 238 исходного файла.

<a id="case_studies"></a>
## 16. Практические кейсы

### RAG-Q118 — Кейс 1. Чат-ассистент на базе LLM с динамическим подбором контекста в зависимости от запроса.

**Теги:** `case_studies`, `applications`, `dynamic_context`
**Источник:** строка 247 исходного файла.

### RAG-Q119 — Кейс 2. Техники промпт-инжиниринга.

**Теги:** `case_studies`, `applications`, `prompting_case_study`
**Источник:** строка 248 исходного файла.
