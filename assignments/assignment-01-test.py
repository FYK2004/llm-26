import json
import re
import spacy
from collections import defaultdict, Counter
nlp = spacy.blank("en")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")
# 读取数据
with open('./assignments/enwiki-train.json','r',encoding='utf-8') as f:
    train_data = [json.loads(line) for line in f]
with open('./assignments/enwiki-test.json','r',encoding='utf-8') as f:
    test_data = [json.loads(line) for line in f]

def process_batch_sentences(texts, batch_size=100):
    """批量处理句子分割，优化内存和速度"""
    sentences_counts = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        sentences_counts.append(len(list(doc.sents)))
    return sentences_counts

print("训练集：")
train_texts_by_class = defaultdict(list)
for doc in train_data:
    train_texts_by_class[doc['label']].append(doc['text'])

for label in sorted(train_texts_by_class.keys()):
    texts = train_texts_by_class[label]
    sentence_counts = process_batch_sentences(texts, batch_size=100)
    avg_sentences = sum(sentence_counts) / len(sentence_counts)
    print(f"类别 '{label}' 的平均句子数: {avg_sentences:.2f}")

print("\n测试集：")
test_texts_by_class = defaultdict(list)
for doc in test_data:
    test_texts_by_class[doc['label']].append(doc['text'])

for label in sorted(test_texts_by_class.keys()):
    texts = test_texts_by_class[label]
    sentence_counts = process_batch_sentences(texts, batch_size=50)
    avg_sentences = sum(sentence_counts) / len(sentence_counts)
    print(f"类别 '{label}' 的平均句子数: {avg_sentences:.2f}")
