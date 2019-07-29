def squeeze_same_results(results):
    squeezed_results = [results[0]]
    for item in results[1:]:
        for squeezed_item in squeezed_results:
            if item['results'] == squeezed_item['results']:
                if item['min_energy'] < squeezed_item['min_energy']:
                    squeezed_item['min_energy'] = item['min_energy']
                    squeezed_item['occurrences'] = item['occurrences']
                elif item['min_energy'] == squeezed_item['min_energy']:
                    squeezed_item['occurrences'] += item['occurrences']
                break
        else:
            squeezed_results.append(item)
    return squeezed_results


def parse_result(response):
    parsed_response = {}
    for datum in response.data(['sample', 'energy', 'num_occurrences']):
        key = (tuple(dict(datum.sample).items()), float(datum.energy))
        if key in parsed_response:
            parsed_response[key] = (datum.sample, parsed_response[key][1] + datum.num_occurrences)
        else:
            parsed_response[key] = (datum.sample, datum.num_occurrences)

    num_runs = sum([parsed_response[key][1] for key in parsed_response])
    results = []
    for key in parsed_response:
        results.append({
            'results': parsed_response[key][0],
            'min_energy': key[1],
            'occurrences': parsed_response[key][1] / num_runs * 100
        })
    return squeeze_same_results(results)


def get_response(response, embedding=None, qubits=None):
    results = parse_result(response)
    if embedding is not None:
        embedding_results = []
        for item in results:
            embedding_item = {
                'results': {},
                'min_energy': item['min_energy'],
                'occurrences': item['occurrences']
            }
            for A in embedding:
                embedding_item['results'][A] = sum([item['results'][qubit] for qubit in embedding[A]])
                embedding_item['results'][A] /= len(embedding[A])
                embedding_item['results'][A] = round(embedding_item['results'][A], 2)
            embedding_results.append(embedding_item)
        results = squeeze_same_results(embedding_results)
    if qubits is not None:
        new_results = []
        for item in results:
            new_results.append({
                'results': {key: item['results'][key] for key in qubits},
                'min_energy': item['min_energy'],
                'occurrences': item['occurrences']
            })
        results = new_results
    return squeeze_same_results(results)


def print_response(response, embedding=None, qubits=None):
    results = get_response(response, embedding=embedding, qubits=qubits)
    for item in results:
        print(item['results'], "Minimum energy: ", item['min_energy'],
              f"Occurrences: {item['occurrences']:.2f}%")


# TODO Need refactoring
def get_response_only_minimal(response, embedding=None, qubits=None):
    results = get_response(response, embedding=embedding, qubits=qubits)
    min_energy = min(map(lambda x: x['min_energy'], results))
    results = list(filter(lambda x: x['min_energy'] == min_energy, results))
    return squeeze_same_results(results)


def print_response_only_minimal(response, embedding=None, qubits=None):
    results_only_minimal = get_response_only_minimal(response, embedding=embedding,
                                                     qubits=qubits)
    for item in results_only_minimal:
        print(item['results'], "Minimum energy: ", item['min_energy'],
              f"Occurrences: {item['occurrences']:.2f}%")
    print()
    total = sum([item['occurrences'] for item in results_only_minimal])
    print(f"Total: {total:.2f}%")
