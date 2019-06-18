from prepro_lib import EmbeddingGenerator, save


if __name__ == "__main__":
    special_words = {'PAD': 0, "OOV": 1}
    g = EmbeddingGenerator(300, special_tokens=special_words)
    emb_info = g.load_word2vec_file('./embedding/Total_word.word', 1292608)

    print("save w2id_dict.pickle")
    save(emb_info.w2id_dict, "./w2id_dict.pickle")

    print("save w2v_dict.pickle")
    save(emb_info.w2v_dict, "./w2v_dict.pickle")

    print("save id2w_dict.pickle")
    save(emb_info.id2w_dict, "./id2w_dict.pickle")

    print("save emb_matrix.pickle")
    save(emb_info.emb_matrix, "./emb_matrix.pickle")
