## Data Preprocessing

Prepare train, validation, and test dataset.


1. Run `preprocess_table.py`
    - Represent table by columns
    - Remove duplicate cell value across column
    - Coverts all value to lower case
    - Replace meaningful symbol with its meaning work, e.g., # -> number, % -> percentage | percent
    - Equip each column with a data type (String, Date, Digit)
    - Remove '"', "'", and '(', ')'
    
2. Run `preprocess_question.py`
    - Tokenize Question
    - Build Ground Truth
   
3. Run `lemmatize.py`
    - Stem all words with PorterStemmer
    - Lemmatize all words with WordNetLemmatizer

4. Run `max_match.py`
    - Use forward-match and backward-match algorithm to lookup table.