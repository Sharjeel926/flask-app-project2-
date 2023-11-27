from flask import Flask, request, jsonify
from process_index_tf_idf import Indexer, Preprocessor, LinkedList  

print("Classes imported successfully")

app = Flask(__name__)


indexer = Indexer()
preprocessor = Preprocessor()

def build_inverted_index(corpus_file):
    inverted_index = {}

    with open(corpus_file, 'r') as file:
        for line in file:
            doc_id, text = preprocessor.get_doc_id(line)
            tokens = preprocessor.tokenizer(text)
            for token in tokens:
                if token not in inverted_index:
                    inverted_index[token] = LinkedList()
                inverted_index[token].insert(doc_id)

    return inverted_index

corpus_file = 'input_corpus.txt'
inverted_index = build_inverted_index(corpus_file)
total_num_docs = len(inverted_index)
print("Total Number of Documents:", total_num_docs)
for term in inverted_index:
    postings_list = inverted_index[term]
    postings_list.idf = total_num_docs / postings_list.get_length()
    postings_list.add_skip_connections()

indexer.inverted_index = inverted_index

@app.route('/execute_query', methods=['POST'])
def execute_query():
    if request.method == 'POST':
        data = request.get_json(force=True)
        queries = data.get('queries')
        results = {}

        for query in queries:
            query_terms = indexer.preprocess_query(query)
            result_docs, comparisons = indexer.daat_and_query(query_terms)

            query_result = {
                "num_comparisons": comparisons,
                "num_docs": len(result_docs),
                "results": result_docs
            }

            results[query] = query_result

        response = {
            "Response": {
                "daatAnd": results,
                "daatAndSkip": results,
            }
        }

        return jsonify(response)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999)
