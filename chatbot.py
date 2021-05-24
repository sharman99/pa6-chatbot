# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util

import numpy as np
import re
from porter_stemmer import PorterStemmer
from collections import Counter
import random

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'moviebot'

        self.creative = creative

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)

        # Create list of user's movie ratings
        self.userRatings = np.zeros(np.shape(ratings)[0])

        #Create counter of ratings by the user and how many movies we have recommended so far
        self.userRated = 0
        self.hasRecommended = 0

        # Save the recommendations calculated between calls of process, so recommend does not need to be called
        # multiple times unnecessarily
        self.recommendedMovies = None

        # Create lists of affirmative, negative, clarification, and recommendation responses
        self.affirmatives = [
            "I see you liked {}. Tell me how you feel about another movie.",
            "A fan of {}, I see. Can you share your opinions about another movie?",
            "{} was a great movie indeed! Can you tell me your feelings about another movie?",
            "You must've watched {} multiple times then! Tell me your feelings about another movie.",
            "I'm a fan of {} too! Tell me how you feel about another movie.",
            "Great taste in movies! I also liked {}. Could you share another movie opinion of yours?",
            "Yes, {} was a great film! Please share your feelings about another movie."
        ]

        self.negatives = [
            "I see you didn't enjoy {} too much. Tell me how you feel about another movie.",
            "Not a huge fan of {}. Can you share your opinions about another movie?",
            "I didn't think {} was too great either. Can you tell me your feelings about another movie?",
            "Yeah, {} could've been better. Tell me your feelings about another movie.",
            "Watching {} once was enough for me, too! Tell me how you feel about another movie.",
            "Great taste in movies! I also didn't like {}. Could you share another movie opinion of yours?",
            "I agree, {} wasn't too great of a film! Please share your feelings about another movie."
        ]

        self.unclear = [
            "I can't tell if you enjoyed {} or not. Could you please clarify?",
            "Could you express your feelings about {} in more detail?",
            "I'm sorry, but I can't quite tell how you feel about {}. Could you further express your sentiment about the movie?",
            "Could you please explain your feelings about {} in more detail? I can't tell if you enjoyed it or not."
        ]

        self.duplicates = [
            "I found several movies called {}. Could you clarify further?",
            "There are multiple movies that go by {}. Could you further detail the movie you are talking about?",
            "Could you please clarify what movie you are talking about? There are multiple movies called {}."
        ]

        self.noFound = [
            "{} doesn't exist in my movie database. Could you please share your opinions on another movie?",
            "Sorry, but I can't find anything about {}. Please share your opinions on another movie.",
            "I've never heard of {}. Could you please express your feelings about another movie?"
        ]

        self.noTitle = [
            "Could you please share your feelings about a movie?",
            "I don't think you shared your feelings about a movie. Could you please do so?",
            "I'm here to discuss movies! Could you please share your feelings about a movie?"
        ]

        self.multipleTitles = [
            "Could you please talk about one movie at a time?",
            "Please only share your feelings about one movie at a time!",
            "I am but a simple bot! Please share your feelings one movie at a time!"
        ]

        self.recommendFirst = [
            "Based on your responses, I think you'd like {}. Would you like more recommendations? (yes/no)",
            "From what I gather, I believe you'd enjoy {}. Would you like more recommendations? (yes/no)"
        ]

        self.recommendMore = [
            "I think you would enjoy {}! Would you like more recommendations? (yes/no)",
            "I believe you would be a big fan of  {}! Would you like to receive more recommendations? (yes/no)",
            "{} would be a great movie for you to watch! Any more recommendations? (yes/no)",
            "{} is right up your alley! Would you like more recommendations? (yes/no)"
        ]

        self.recommendLast =[
            "My final recommendation for you is {}!",
            "The last movie I can recommend to you is {}!",
            "I think you'd like {}! That's my last recommendation!"
        ]
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "How can I help you?"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Have a nice day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        if self.creative:
            response = "I processed {} in creative mode!!".format(line)
        else:
            # Check if we are in the recommendation phase or not
            if self.userRated < 5:
                titles = self.extract_titles(line)
                # Ensure the user input has exactly one title, and that the title is clarifying enough
                if len(titles) == 0:
                    return random.choice(self.noTitle)
                elif len(titles) > 1:
                    return random.choice(self.multipleTitles)
                userTitle = titles[0]
                titleIndices = self.find_movies_by_title(userTitle)
                if len(titleIndices) == 0:
                    return random.choice(self.noFound).format(userTitle)
                elif len(titleIndices) > 1:
                    return random.choice(self.duplicates).format(userTitle)
                # Find that movie in the database, and respond based on the acquired sentiment of the user input.
                movieIndex = titleIndices[0]
                movieTitle = self.titles[titleIndices[0]][0]
                sentiment = self.extract_sentiment(line)
                if sentiment == 0:
                    return random.choice(self.unclear).format(movieTitle)
                else:
                    self.userRated += 1
                    if sentiment == 1:
                        self.userRatings[movieIndex] = 1
                    else:
                        self.userRatings[movieIndex] = -1
                    # Respond based on whether or not it is time to recommend movies
                    if self.userRated < 5:
                        if sentiment == 1:
                            return random.choice(self.affirmatives).format(movieTitle)
                        else:
                            return random.choice(self.negatives).format(movieTitle)
                    else:
                        self.recommendedMovies = self.recommend(self.userRatings, self.ratings)
                        self.hasRecommended += 1
                        recMovieIndex = self.recommendedMovies[self.hasRecommended - 1]
                        recTitle = self.titles[recMovieIndex][0]
                        return random.choice(self.recommendFirst).format(recTitle)
            else:
                # In the recommendation phase, respond based on the user input and how many movies we have recommended
                if self.hasRecommended >= len(self.recommendedMovies):
                    return "I am out of movie recommendations! Type :quit when you are done!"
                elif line.lower() == "yes":
                    self.hasRecommended += 1
                    if self.hasRecommended == len(self.recommendedMovies):
                        recMovieIndex = self.recommendedMovies[self.hasRecommended - 1]
                        recTitle = self.titles[recMovieIndex][0]
                        return random.choice(self.recommendLast).format(recTitle)
                    else:
                        recMovieIndex = self.recommendedMovies[self.hasRecommended - 1]
                        recTitle = self.titles[recMovieIndex][0]
                        return random.choice(self.recommendMore).format(recTitle)
                elif line.lower() == "no":
                    return "Ok! I'll just wait here! Type :quit to leave, or type \'yes\' to get another recommendation!"
                else:
                    return "If you want a recommendation, type \'yes\' exactly. Otherwise, type :quit if you are done!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, and extract_sentiment_for_movies
        methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        return re.findall('"([^"]*)"', preprocessed_input)

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        if title.find("An") == 0:
            title = title.replace("An ", "") 
            i = title.find("(")
            if i != -1:
                title = title[0:i-1] + ", An " + title[i:]
            else:
                title = title + ", An"
        elif title.find("The") == 0:
            title = title.replace("The ", "")
            i = title.find("(")
            if i != -1:
                title = title[0:i-1] + ", The " + title[i:]
            else:
                title = title + ", The"
        elif title.find("A") == 0:
            title = title.replace("A ", "") 
            i = title.find("(")
            if i != -1:
                title = title[0:i-1] + ", A " + title[i:]
            else:
                title = title + ", A"
        
        return_list = []
        movies = open("./data/movies.txt", "r")
        for line in movies:
            line_list = line.split("%")
            if title.find("(") != -1:
                if line_list[1] == (title):
                    return_list.append(int(line_list[0]))
            else:
                if line_list[1].find(title + " (") != -1:
                    return_list.append(int(line_list[0]))
        return return_list

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        stemmer = PorterStemmer()

        #stem sentiment.txt
        sentimentDict = {}
        for key in self.sentiment:
            new_key = stemmer.stem(key, 0, len(key) - 1)
            sentimentDict[new_key] = self.sentiment[key]
        
        s = re.sub('"([^"]*)"', '', preprocessed_input).lower()
        #s = re.sub(r'[^\w\s]','', s).lower()
        tokens  = s.split(" ")

        posCount = 0
        negCount = 0
        switch = False
        reverse_switch = False

        for token in tokens:
            token = stemmer.stem(token, 0, len(token) - 1)
            if token.find(",") != -1:
                token = re.sub(',','', token)
                reverse_switch = True
            if token in sentimentDict:
                if sentimentDict[token] == "pos":
                    if switch:
                        negCount += 1
                    else:
                        posCount += 1
                else:
                    if switch:
                        posCount += 1
                    else:
                        negCount += 1
            
            if reverse_switch:
                switch = False

            if token == "didn't" or token == "never" or token == "not":
                switch = True
        
        if negCount > posCount:
            return -1
        elif posCount > negCount:
            return 1
        else:
            return 0

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of
        pre-processed text that may contain multiple movies. Note that the
        sentiments toward the movies may be different.

        You should use the same sentiment values as extract_sentiment, described

        above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess(
                           'I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie
        title, and the second is the sentiment in the text toward that movie
        """
        pass

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least
        edit distance from the provided title, and with edit distance at most
        max_distance.

        - If no movies have titles within max_distance of the provided title,
        return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given
        title than all other movies, return a 1-element list containing its
        index.
        - If there is a tie for closest movie, return a list with the indices
        of all movies tying for minimum edit distance to the given movie.

        Example:
          # should return [1656]
          chatbot.find_movies_closest_to_title("Sleeping Beaty")

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title
        and within edit distance max_distance
        """

        pass

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (eg. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it
        should return a list with the indices it could be referring to (to
        continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        pass

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.zeros_like(ratings)
        #go over each element, and only modify those that are not 0 already
        rows, cols = np.shape(ratings)
        for i in range(rows):
            for j in range(cols):
                if ratings[i, j] > threshold:
                    binarized_ratings[i, j] = 1
                elif ratings[i, j] != 0:
                    binarized_ratings[i, j] = -1

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        dot_prod = np.dot(u, v)
        denom = np.linalg.norm(u, ord=2) * np.linalg.norm(v, ord=2)
        similarity = dot_prod / denom
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For starter mode, you should use item-item collaborative filtering   #
        # with cosine similarity, no mean-centering, and no normalization of   #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        ranks = {}
        has_rated = [i for i in range(len(user_ratings)) if user_ratings[i] != 0] #list of indices already rated by user
        for movie in range(len(user_ratings)):
            if movie in has_rated:
                continue
            rating = 0
            for rated_movie in has_rated:
                rating += self.similarity(ratings_matrix[movie], ratings_matrix[rated_movie]) * user_ratings[rated_movie]
            ranks[movie] = rating
        sortedRatings = Counter(ranks)
        recommendations = [elem[0] for elem in sortedRatings.most_common(k)]

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return """
        Your task is to implement the chatbot as detailed in the PA6
        instructions.
        Remember: in the starter mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
