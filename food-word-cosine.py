import parser, numpy, numpy.linalg, copy

def compute_centroid(terms_to_vecs, vocab):
    assert len(vocab) >= 1
    centroid_vec = copy.deepcopy(terms_to_vecs[vocab[0]])
    n = 1
    for term in vocab[1:]:
        if term not in terms_to_vecs:
            continue
        term_vec = terms_to_vecs[term]
        for i, dim in enumerate(term_vec):
            centroid_vec[i] = (term_vec[i] + n*centroid_vec[i]) / (n+1)
        n += 1
    return centroid_vec

def get_similar_terms(terms_to_vecs, vocab):
    centroid = compute_centroid(terms_to_vecs, list(vocab))
    similar = set()
    for term, vec in terms_to_vecs.items():
        similarity = numpy.vdot(centroid, vec)
        if similarity >= 0.59:
            similar.add(term)
    return similar

def main():
    terms_to_vecs = {}
    healthy_foods, neutral_foods, unhealthy_foods = parser.get_food_scores("food_scores.txt")
    with open("glove.twitter.27B.25d.txt", encoding='utf8') as vector_file:
        for line in vector_file:
            tokens = line.strip().split()
            if len(tokens) <= 2:
                continue
            term = tokens[0]
            vec = numpy.array(tokens[1:], dtype=float)
            if len(vec) != 25:
                print("Error: Wrong vec length:", term)
                continue
            terms_to_vecs[term] = vec

    print("Normalizing vectors...")
    for term, vec in terms_to_vecs.items():
        norm = numpy.linalg.norm(vec)
        if norm == 0:
            norm = numpy.finfo(vec.dtype).eps
        vec /= norm

    print("Computing cosine similarities...")
    close_to_healthy = get_similar_terms(terms_to_vecs, healthy_foods)
    close_to_neutral = get_similar_terms(terms_to_vecs, neutral_foods)
    close_to_unhealthy = get_similar_terms(terms_to_vecs, unhealthy_foods)

    close_to_healthy.difference_update(close_to_neutral, close_to_unhealthy, \
                                       unhealthy_foods, \
                                       neutral_foods, healthy_foods)
    close_to_neutral.difference_update(close_to_healthy, close_to_unhealthy, \
                                       unhealthy_foods, \
                                       neutral_foods, healthy_foods)
    close_to_unhealthy.difference_update(close_to_neutral, close_to_healthy, \
                                         unhealthy_foods, \
                                         neutral_foods, healthy_foods)

    print("Close to healthy:")
    for term in sorted(close_to_healthy):
        print(term)
    print()

    print("Close to neutral:")
    for term in sorted(close_to_neutral):
        print(term)
    print()

    print("Close to unhealthy:")
    for term in sorted(close_to_unhealthy):
        print(term)
    print()

main()
