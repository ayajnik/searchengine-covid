# maximum number of articles you want returned for each query
TOP_N_ARTICLES = 5

# maximum number of points/sentences you want returned for each article summary 
TOP_N_POINTS = 5
import time
import re
import json
import pickle
import re

from nltk.corpus import stopwords
import os
import pandas as pd

from collections import Counter

def is_match(text, keywords):
    '''Check if any `keywords` exist in `text`.

    Parameters
    ----------
    text : str
    keywords : List[str]

    Returns
    -------
    bool
    '''
    return any(f'{keyterm} ' in text.lower() for keyterm in keywords)


def clean_text(text):
    '''Remove punctuation from query and lowercase letters.

    Parameters
    ----------
    query : str

    Returns
    -------
    str
    '''
    return re.sub('\?|!|\.|,|\(|\)', '', text).lower()

def get_data_corpus():
    '''Get corpus data.
    
    Returns
    -------
    str
    '''
    data = pd.read_csv(ARTICLE_INFO_PATH).loc[:, 'text']
    corpus_text = ' '.join(data.values)
    return corpus_text.replace("\n", " ")


def get_stopwords_from_corpus(corpus):
    '''Extract stopwords from corpus based on word frequency.
    
    Parameters
    ----------
    corpus : str
    
    Returns
    -------
    List[str]
    '''
    sample = corpus.lower().split(' ')

    word_counts = Counter(sample)

    max_word_count = max(word_counts.values())

    threshold = 5000
    return [word for word, count in word_counts.items() if count > threshold]


CORPUS_STOPWORDS = [
    'number',
    'human',
    'cases',
    'also',
    'reported',
    'one',
    'immune',
    'response',
    'within',
    'influenza',
    'among',
    'different',
    'high',
    'found',
    'showed',
    'use',
    'identified',
    'two',
    'used',
    'results',
    'analysis',
    'performed',
    'using',
    'described',
    'detected',
    'including',
    'group',
    'could',
    'observed',
    'significant',
    'based',
    'shown',
    'however,',
    'compared',
    'higher',
    'may',
    'specific',
    'studies',
    'study',
    'type',
    'well',
    'although',
    'levels',
    'host',
    'activity',
    'data',
    'associated',
    'due',
    'samples',
    'figure',
    'table',
    'case',
    'effect', 
    'effects', 
    'affected',
    'across',
    'within',
    'humans',
    'who',
    'what',
    'why',
    'how',
    'distribution',
    'eg',
    'ie',
    'prevalence',
    'particularly',
    'whether',
    'make',
    'even',
    'might',
    '2019',
]
COVID_19_TERMS = [
    'covid-19', 
    'covid 19',
    'covid-2019',
    '2019 novel coronavirus', 
    'corona virus disease 2019',
    'coronavirus disease 19',
    'coronavirus 2019',
    '2019-ncov',
    'ncov-2019', 
    'wuhan virus',
    'wuhan coronavirus',
    'wuhan pneumonia',
    'NCIP',
    'sars-cov-2',
    'sars-cov2',
]

VIRUS_TERMS = [
    'epidemic', 
    'pandemic', 
    'viral',
    'virus',
    'viruses',
    'coronavirus', 
    'respiratory',
    'infectious',
] + COVID_19_TERMS

from nltk.corpus import stopwords
    

def get_stopwords():
    '''Get english stopwords and corpus stopwords.''' 
    return set(VIRUS_TERMS + CORPUS_STOPWORDS + stopwords.words('english'))

import re

# url links, 'doi preprint', [citations]
ARTIFACTS = r'https?:\/\/.[^\s\\]*|doi: medRxiv|preprint|\[\d+\]|\[\d+\, \d+\]'


class Article():
    '''`Article` object for storing article text information.'''

    def __init__(self, article):
        '''Initialize `Article` object.'''
        self.article = article
        
    def get_title(self):
        '''Article title.'''
        return self.article['metadata']['title']
    
    def get_abstract(self):
        '''Article abstract bodytext.'''
        return self.clean_text_of_artifacts(
            self.combine_bodytext(self.article.get('abstract', []))
        )

    def get_bodytext(self):
        '''Article main text.'''
        return self.clean_text_of_artifacts(
            self.combine_bodytext(self.article.get('body_text', []))
        )

    def get_full_text(self):
        '''Article abstract and body text.'''
        return self.get_abstract() + ' ' + self.get_bodytext()
    
    @staticmethod
    def clean_text_of_artifacts(text):
        '''Remove URL links and other artifacts from text.
        
        Parameters
        ----------
        text : str

        Returns
        -------
        str
        '''
        return re.sub(ARTIFACTS, '', text, flags=re.MULTILINE)

    @staticmethod
    def combine_bodytext(text_info):
        '''Get combined text fields from list of dicts.
        
        Parameters
        ----------
        text_info : List[Dict[str]]
            List of body text.

        Returns
        -------
        str
            `text_info` joined together into string.
        '''
        return ' '.join(x['text'] for x in text_info)

def filter_covid19_articles(df):
    '''Filter DataFrame on articles that contain COVID-19 keyterms.

    Parameters
    ----------
    df : pandas.DataFrame
        Article info, including 'text' column.

    Returns
    -------
    pandas.DataFrame
        Article info of COVID-19 related papers.
    '''
    return df[
        df['text'].apply(lambda x: is_match(x, set(COVID_19_TERMS)))
    ]


def tokenize_documents(corpus):
    '''Tokenize corpus of documents.
    
    Parameters
    ----------
    corpus : List[str]
        Corpus of research paper documents.
    
    Returns
    -------
    List[List[str]]
        documents --> words
    '''
    return [
        [
            word for word in word_tokenize(clean_text(doc)) 
            if word not in get_stopwords()
        ] for doc in corpus 
    ] 

def load_model(filename):
    '''Load pickled model.'''
    return pickle.load(open(filename, 'rb'))

def train_bm25_model(corpus):
    '''Train an Okapi BM25 model on corpus of research articles.

    Parameters
    ----------
    corpus : pandas.Series

    Returns 
    -------
    rankbm25.BM25Okapi
        Okapi BM25 model trained on corpus data.
    '''    
    logging.info('Tokenizing documents...')
    tokenized_corpus = tokenize_documents(corpus)
        
    logging.info('Training BM25 model...')
    return BM25Okapi(tokenized_corpus)

start_time = time.time()

# main()
bm25_model = load_model("E:\\COVID-app\\bm25_model")
    
seconds = time.time() - start_time
minutes = seconds / 60
print('Took {:.2f} minutes'.format(minutes))

from gensim.models import KeyedVectors, Word2Vec 

class SearchQuery():
    '''`SearchQuery` object for cleaning and processing a `query` input.'''

    SIMILARITY_THRESHOLD = 0.62

    def __init__(self, query):
        '''Initialize `SearchQuery` object.
        
        Parameters
        ----------
        query : str
        '''
        self.query = clean_text(query)
        self.init_query_keywords()
        self.init_related_keywords(self.get_word2vec_model())

    def init_query_keywords(self):
        '''Initialize query keywords.'''
        self.query_keywords = [
            x for x in self.query.split() if x not in get_stopwords()
        ]

    def init_related_keywords(self, word2vec_model):
        '''Initialize keywords related to `query_keywords`.

        Iterates over each keyterm in `query_keywords` and finds related words 
        from the trained `Word2Vec` vocabulary. If there's a high enough 
        similarity score, adds it to `related_keywords`.
        
        Parameters
        ----------
        word2vec_model : gensim.models.Word2Vec
            `Word2Vec` model trained on corpus data.  
        '''
        self.related_keywords = []
        for word in self.query_keywords:
            if word in word2vec_model.wv.vocab:
                self.related_keywords += [
                    x[0] for x in word2vec_model.wv.most_similar(word, topn=10) 
                    if x[1] > SearchQuery.SIMILARITY_THRESHOLD
                ]
    
    @staticmethod
    def get_word2vec_model():
        '''Load `Word2Vec` model previously trained on the dataset.
        
        Returns
        -------
        gensim.models.Word2Vec
        '''
        return Word2Vec.load("E:\\COVID-app\\word2vec_model")

import re

from gensim.summarization.summarizer import summarize


class Summary():
    '''`Summary` object for extracting executive summary from text.'''
    
    def __init__(self, text, query_keywords):
        '''Initialize `Summary` object.
        
        Parameters
        ----------
        text : str
        query_keywords : List[str]
        '''
        self.text = text
        self.keywords = query_keywords

    def get_topn_sentences(self):
        '''Get top `n` sentences of text as summary.
        
        Returns
        -------
        List[str]
        '''
        ranked_sentences = summarize(self.text, split=True) 
        relevant_sentences = self.filter_relevant_sentences(ranked_sentences)

        return relevant_sentences[:TOP_N_POINTS]
    
    @staticmethod
    def is_decimal_value_in_text(text):
        '''Check if there is a decimal value or percentage within the text.

        Make sure that decimal value is not a Figure or Section number.

        Parameters
        ----------
        text : str

        Returns
        -------
        bool
        '''
        patterns = [
            r'(?<!Section )([0-9]+\.[0-9]+|%)',
            r'(?<!SECTION )([0-9]+\.[0-9]+|%)',
            r'(?<!Figure )([0-9]+\.[0-9]+|%)',
            r'(?<!FIGURE )([0-9]+\.[0-9]+|%)',
            r'(?<!Fig )([0-9]+\.[0-9]+|%)',
            r'(?<!Fig. )([0-9]+\.[0-9]+|%)',
            r'(?<!Tables )([0-9]+\.[0-9]+|%)',
            r'(?<!Chapter )([0-9]+\.[0-9]+|%)',
            r'(?<!CHAPTER )([0-9]+\.[0-9]+|%)',
        ]

        if all(re.search(pattern, text) for pattern in patterns):
            return True 
        
        return False
    def filter_relevant_sentences(self, sentences):
        '''Filter sentences on relevancy filter. 
        
        If filters out all sentences, returns original unfiltered sentences instead.
        
        NOTE: Previously was filtering on whether keyword exists in sentence. Now
        filters on whether decimal value exists in sentence.

        Parameters
        ----------
        sentences : List[str]

        Returns
        -------
        List[str]
        '''
        filtered_sentences = [
            sentence for sentence in sentences 
            if self.is_decimal_value_in_text(sentence)
        ]
        
        if not filtered_sentences:
            return sentences
        
        return filtered_sentences

class SearchResult():
    '''`SearchResult` object for storing search result article information.'''

    def __init__(self, title, text, url, query_keywords):
        '''Initialize `SearchResult` object.

        Parameters
        ----------
        title : str
            Article title.
        text : str
            Article text.
        url : str
            Article url link.
        query_keywords: List[str]
            Query search keywords.
        '''
        self.title = title
        self.text = text
        self.url = url
        self.keywords = query_keywords 

        self.main_points = self.get_topn_points()

        if isinstance(title, str):
            text = text + title
        
        self.study_info = StudyInfo(text, url)
    
    def get_topn_points(self):
        '''Keep `n` most highly ranked article points.'''
        points = Summary(self.text, self.keywords)
        
        return points.get_topn_sentences()

class StudyInfo():
    '''Object for extracting the level of evidence for findings in paper.'''

    def __init__(self, article_text, url_link):
        '''Initialize `StudyInfo` object.

        Parameters
        ----------
        article_text : str
        url_link : str
        '''
        self.article_text = article_text
        self.url = url_link
        self.peer_reviewed = self.is_peer_reviewed(article_text)
        self.num_studies = self.extract_number_of_studies(article_text)
        self.sample_size = self.extract_sample_size(article_text)
        self.study_designs = self.extract_study_design(article_text) 

    @staticmethod
    def is_peer_reviewed(text):
        '''Check if paper is peer-reviewed.

        Returns
        -------
        Optional[bool]
            Returns None if unsure.
        '''
        non_peer_reviewed_clause = 'was not peer-reviewed'

        # "PMC does not include any non peer-reviewed research articles."
        peer_review_terms = {
            'peer-reviewed', 
            'peer reviewed', 
            'peer review', 
            'pubmed', 
            'ncbi', 
            'pmc',
        }
        if non_peer_reviewed_clause in text:
            return False
        elif is_match(text, peer_review_terms):
            return True

        return None

    @staticmethod
    def extract_number_of_studies(text):
        '''Extract the number of studies performed in article research.

        Does so by searching for the term 'studies' and returning the numeric 
        value right before it.
        
        Parameters
        ----------
        text : str

        Returns
        -------
        Optional[int]
            Returns None if no match found.
        '''
        pattern = r'(?:([0-9])[a-zA-Z ]{0,5}(?:studies))'

        m = re.search(pattern, text)
        if m:
            return int(m.group(1))
    
    @staticmethod
    def extract_sample_size(text):
        '''Extract the sample size of the article research.

        Does so by searching for the term 'sample size of' and returning the 
        numeric value right after it.
        
        Parameters
        ----------
        text : str

        Returns
        -------
        Optional[int]
            Returns None if no match found.
        '''
        pattern1 = r'(?:total sample size of( about| over)?)(.[0-9,]+)'
        pattern2 = r'(?:sample size of( about| over)?)(.[0-9,]+)'
        pattern3 = r'(.[0-9,]+)(.{,14})(?: patients| participants)'

        m1 = re.search(pattern1, text)
        m2 = re.search(pattern2, text)
        m3 = re.search(pattern3, text)
        value = None
        if m1:
            value = m1.group(2).replace(',', '')
        elif m2:
            value = m2.group(2).replace(',', '')
        elif m3:
            value = m3.group(1).replace(',', '')

        # 'SARS-2 patients' returns -2
        # 'COVID-19 patients' returns -19
        # added to prevent this
        if value:
            try:
                if int(value) > 0:
                    return int(value)
            except:
                return None

    @staticmethod
    def extract_study_design(text):
        
        '''Extracts the type of study design in paper.
        
        Parameters
        ----------
        text : str
        
        Returns
        -------
        List[str]
        '''
        study_designs = [
            'case control',
            'case study',
            'cross sectional',
            'cross-sectional',
            'descriptive study',
            'ecological regression',
            'experimental study',
            'meta-analysis',
            'non-randomized',
            'non-randomized experimental study',
            'observational study',
            'prospective case-control',
            'prospective cohort',
            'prospective study',
            'randomized',
            'randomized experimental study',
            'retrospective cohort',
            'retrospective study',
            'simulation', 
            'systematic review',
            'time series analysis',    
        ]
        
        return [design for design in study_designs if design in text]

class ResultsHTMLText():
    '''Object for storing search results in HTML text template format.''' 

    SPACES = '&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp;'
    ARTICLE_LINK_HEADER = '<a href="{}"> <i>{}</i></a><br>'
    ARTICLE_HEADER = '<i>{}</i> <br>'

    HIGHLIGHT = '<span style="background-color:#FFB6C1">{}</span>'

    def __init__(self, results_info):
        '''Initizialize `ResrultsHTMLText` object.
        
        Parameters
        ----------
        results_info : List[SearchResult]
        '''
        self.results = results_info
        self.results_text = ''

    def print_peer_review_status(self, peer_reviewed):
        '''Add information to on whether the paper had been peer-reviewed.

        Parameters
        ----------
        peer_reviewed : bool
        '''
        if peer_reviewed:
            self.results_text += (
                '&#9830; This paper has been peer-reviewed.<br>'
            )
        elif peer_reviewed is False:
            self.results_text += (
                '&#9830; This paper has NOT been peer-reviewed.<br>'
            )

    def print_num_studies_info(self, num_studies):
        '''Add information on the number of studies in the research.

        Parameters
        ----------
        num_studies : Union[int, str]
        '''
        if num_studies:
            self.results_text += (
                f'&#9830; number of studies: {num_studies}<br>'
            )
    
    def print_sample_size_info(self, sample_size):
        '''Add information on the sample size of the paper study.

        Parameters
        ----------
        sample_size : Union[int, str]
        '''
        if sample_size:
            self.results_text += f'&#9830; sample size: {sample_size}<br>'

    def print_study_design_info(self, design):
        '''Add information of the study design type.

        Parameters
        ----------
        design : List[str]
        '''
        if design:
            self.results_text += (
                f"&#9830; study design: {', '.join(design)}<br>"
            )
            
    def get_results_text(self):
        '''Get results in HTML template format.

        Returns
        -------
        str
        '''
        if not self.results:
            return self.get_no_search_results_found_text()

        self.results_text += '<br>'
        for result in self.results:
            if result.main_points:  
                if isinstance(result.title, float):
                    result.title = 'Title Unknown'

                if isinstance(result.url, float):
                    self.results_text += self.ARTICLE_HEADER.format(
                        result.title
                    )
                else:
                    self.results_text += self.ARTICLE_LINK_HEADER.format(
                        result.url, result.title
                    )
                
                info = result.study_info
                self.print_peer_review_status(info.peer_reviewed)
                # self.print_num_studies_info(info.num_studies)
                self.print_sample_size_info(info.sample_size)
                self.print_study_design_info(info.study_designs)
                self.add_article_mainpoints_text(result) 
                self.results_text += '<br>'    

        return self.results_text
    def add_article_mainpoints_text(self, result):
        '''Return text of main points within the article, in bullet format.
        
        Parameters
        ----------
        search_result : SearchResult
        '''
        self.results_text += '<p>'
        for point in result.main_points:    
            words = [
                ResultsHTMLText.HIGHLIGHT.format(word) 
                if word.replace(',', '') in result.keywords else word 
                for word in point.split()
            ]

            point = ' '.join(words)
            self.results_text += '{} -- {} <br><br>'.format(self.SPACES, point)
    
        self.results_text += '</p>'

    def get_no_search_results_found_text(self):
        '''Return text informing user no results were found.'''
        return (
            'No results found -- It appears not a lot of scientific research '
            'has been done in this area.'
        )

    
    
class ResultsDataFrame():
    '''Object for storing search results as pandas.DataFrame.''' 

    SPACES = '&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp;'
    ARTICLE_LINK_HEADER = '<a href="{}"> <i>{}</i></a><br>'
    ARTICLE_HEADER = '<i>{}</i> <br>'

    HIGHLIGHT = '<span style="background-color:#FFB6C1">{}</span>'

    COLS = [
        'Title', 'Article is peer reviewed', 'Sample size of Article', 'Study Design', 'Main Points'
    ]
    
    def __init__(self, results_info):
        '''Initizialize `ResrultsHTMLText` object.
        
        Parameters
        ----------
        results_info : List[SearchResult]
        '''
        self.results = results_info
        self.results_df = pd.DataFrame(columns=self.COLS)
        
    def get_results_df(self):
        '''Get results in pandas.DataFrame.

        Returns
        -------
        pandas.DataFrame
        '''
        for result in self.results:
            
            row = []
            if result.main_points:  
                if isinstance(result.title, float):
                    result.title = 'Title Unknown'

                if isinstance(result.url, float):
                    row.append(self.ARTICLE_HEADER.format(result.title))
                else:
                    row.append(
                        self.ARTICLE_LINK_HEADER.format(
                                result.url, result.title
                        )
                    )
                
                info = result.study_info
                row.append(info.peer_reviewed)
                # row.append(info.num_studies)
                row.append(info.sample_size)
                row.append(', '.join(info.study_designs))
                row.append(self.get_article_mainpoints_text(result))
            
            row_df = pd.DataFrame([row], columns=self.COLS)
            self.results_df = pd.concat([self.results_df, row_df], ignore_index=True)

        return self.results_df
    
    def get_article_mainpoints_text(self, result):
        '''Return text of main points within the article, in bullet format.
        
        Parameters
        ----------
        search_result : SearchResult
        
        Returns
        -------
        str
        '''
        text = ''
        for point in result.main_points:    
            words = [
                self.HIGHLIGHT.format(word) 
                # if word.replace(',', '') in result.keywords else word 
                if Summary.is_decimal_value_in_text(word) else word
                for word in point.split()
            ]

            point = ' '.join(words)
            text += '{} -- {} <br><br>'.format(self.SPACES, point)
        return text

    
        
import pandas as pd


SEARCH_SCORE_THRESHOLD = 10


def get_n(bm25_model, keywords):
    '''Get number of articles that pass threshold.

    NOTE: counts only the articles in the `TOP_N_ARTICLES`.

    Parameters
    ----------
    bm25_model : rank_bm25.BM25Okapi
        Ranking/scoring model trained on corpus dataset.
    keywords : List[str]
        Search query keywords.

    Returns
    -------
    int
        Number of similarity scores of `top_n` articles that pass threshold.
    '''
    return len(
        [score for score in sorted(
            bm25_model.get_scores(keywords), reverse=True
        )[:TOP_N_ARTICLES] if score > SEARCH_SCORE_THRESHOLD]
    )

def get_search_results(search):
    
    global data
    global num_pass_threshold
    global all_keywords
    
    '''Get search results of search query.

    Parameters
    ----------
    search : SearchQuery

    Returns
    -------
    List[SearchResult]
    '''
    data = pd.read_csv("E:\\COVID-app\\article_info.csv")
    
    all_keywords = search.query_keywords + search.related_keywords
    
    bm25_model = load_model("E:\\COVID-app\\bm25_model")
    num_pass_threshold = get_n(bm25_model, all_keywords)

    if num_pass_threshold == 0:
        return []
    else:
        if num_pass_threshold != 0:

            results_text = bm25_model.get_top_n(
                all_keywords, data['text'], n = num_pass_threshold
            )
            results_title = bm25_model.get_top_n(
                all_keywords, data['title_meta'], n =num_pass_threshold
            )

            results_url = bm25_model.get_top_n(
                all_keywords, data['url'], n =num_pass_threshold
            )

            return [
                SearchResult(
                    title, text, url, all_keywords
                ) for title, text, url in zip(results_title, results_text, results_url)
            ]

def get_query_results(query):
    '''Get results of search query.
    
    Parameters
    ----------
    query : str
    
    Returns
    -------
    ResultsHTMLText
    '''
    
    global keywords
    
    search_query = SearchQuery(query)
    
    keywords = search_query.query_keywords + search_query.related_keywords
    display(HTML(f'<h3>Search Terms: {", ".join(keywords)}</h3>'))

    results = get_search_results(search_query)
    # return ResultsHTMLText(results).get_results_text()
    return ResultsDataFrame(results).get_results_df()

from IPython.core.display import display, HTML

def print_answers(queries):
    '''Print search results to each query.
    
    Parameters
    ----------
    tasks : List[str]
    '''
    for query in queries:
        display(HTML(f'<h2>{query} \n</h2>'))
        final_result = get_query_results(query)

        # display(HTML(final_result))
        display(final_result.style)
        


