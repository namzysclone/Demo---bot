from sentence_transformers import SentenceTransformer, util

# Load model once
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_best_column_match(query, possible_columns):
    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    column_embeddings = embed_model.encode(possible_columns, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, column_embeddings)[0]
    best_match_idx = scores.argmax().item()
    return possible_columns[best_match_idx], scores[best_match_idx].item()

def get_best_value_match(value, column_values):
    if not column_values:
        return None, 0
    value_embedding = embed_model.encode(value, convert_to_tensor=True)
    column_embeddings = embed_model.encode(column_values, convert_to_tensor=True)
    scores = util.cos_sim(value_embedding, column_embeddings)[0]
    best_match_idx = scores.argmax().item()
    return column_values[best_match_idx], scores[best_match_idx].item()

def infer_contextual_filters(user_query, df, column_mapping):
    filters = {}
    words = user_query.lower().split()
    key_phrases = [
        ("by", "Brand"),
        ("via", "Channel"),
        ("from", "Country"),
        ("on", "Platform"),
        ("through", "Channel"),
        ("using", "Media"),
        ("in", "Sector"),
        ("under", "Category")
    ]

    for keyword, logical_col in key_phrases:
        if keyword in words:
            try:
                keyword_index = words.index(keyword)
                candidate_word = words[keyword_index + 1] if keyword_index + 1 < len(words) else None
                if not candidate_word:
                    continue

                actual_col = column_mapping.get(logical_col)
                if not actual_col or actual_col not in df.columns:
                    continue

                column_values = df[actual_col].dropna().astype(str).unique().tolist()
                match, score = get_best_value_match(candidate_word, column_values)
                if match and score > 0.5:
                    filters[actual_col] = match
            except Exception as e:
                print(f"Context filter error on '{keyword}': {e}")
                continue

    return filters

# 🔁 PATCH THIS INTO YOUR QUERY HANDLER:
# Example usage:
# user_query = st.session_state.chat_input
# filters = extract_filters(user_query, df, st.session_state.column_mapping)
# contextual_filters = infer_contextual_filters(user_query, df, st.session_state.column_mapping)
# filters.update(contextual_filters)
