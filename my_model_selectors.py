import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError



    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        # best_num_state = None
        alpha = 1.5
        best_bic_score = float('+inf')
        best_model = None
        for num_state in range(self.min_n_components,self.max_n_components+1):

            try:
                hmm_model = GaussianHMM(n_components=num_state, covariance_type="diag", n_iter=1000,
                    random_state=self.random_state, verbose=False).fit(self.X,self.lengths)

                logL = hmm_model.score(self.X,self.lengths)
                n_parameters = num_state * num_state + 2 * num_state * len(self.X[0]) -1
                bic_score = -2 * logL + alpha * n_parameters * np.log(len(self.X))

                # print("Training {} with {} states. BIC score: {}".format(self.this_word, num_state, bic_score))
                if bic_score < best_bic_score:
                    best_bic_score = bic_score
                    best_model = hmm_model
                    # print("BestModel for {} set to {} states".format(self.this_word,num_state))

            except:
                pass


        



        return best_model if best_model is not None else self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def logL_anti_likelihood_terms(self, model):
        # print("LogL anti likelihood terms")
        # print("all words:{}".format(self.words))

        # antilogLikelihoodArr=[]

        antiLL = 0.0
        words_counter = 0
        for word, features in self.hwords.items():
            if word != self.this_word:
                X, lengths = features
                # print("Current word detected:{} value: {}".format(word,features))
                # print("antilog X len:{} , len lengths : {} for {}".format(len(X), len(lengths), word))

                #Train model
                try:
                    # model = GaussianHMM(n_components = num_state, covariance_type = "diag", n_iter=1000,
                    #     random_state= self.random_state, verbose=False).fit(X, lengths)

                    logL = model.score(X,lengths)


                    # print("Antiloglikelihood logL:{}".format(logL))

                    # if logL is not None:
                    antiLL += logL
                    words_counter+=1
                        # antilogLikelihoodArr.append(logL)

                except Exception as e: 
                    print("Error: {}".format(e))
                    pass


        #get Average of logL
        # print("antiloglikelihoodArr:{}".format(antilogLikelihoodArr))
        # avgAntiLogLikelihood = np.mean(antilogLikelihoodArr)
        # print("AvgAntiLogLikelihood: {}".format(avgAntiLogLikelihood))


        return antiLL / words_counter




    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)


        best_dic_score = float('-inf')
        best_model = None

        # print("len Self.X :{}\n len Self.lengths:{} for {}".
        #     format(len(self.X),len(self.lengths),self.this_word))
        


        #self.min_n_components, self.max_n_components +1 
        for num_state in range(self.min_n_components, self.max_n_components +1):
            try:
                hmm_model = GaussianHMM(n_components = num_state, covariance_type = "diag", n_iter=1000,
                    random_state= self.random_state, verbose=False).fit(self.X, self.lengths)

                logL = hmm_model.score(self.X, self.lengths)
                # print("LogL score for {} : {} ".format(self.this_word, logL))
                antiLogL = self.logL_anti_likelihood_terms(hmm_model)
                dic_score = logL - antiLogL
                # print("logL:{}, avg antiLogL :{} , score:{} for {} for {} components".
                #     format(logL,antiLogL,dic_score, self.this_word,num_state))


                # if best_dic_score == None:
                #     best_dic_score = dic_score
                #     best_model = hmm_model

                if dic_score > best_dic_score:
                    best_dic_score = dic_score
                    best_model = hmm_model

            except Exception as e: 
                # print("Error:{}".format(e))
                pass

        # TODO implement model selection based on DIC scores
        return best_model if best_model is not None else self.base_model(self.n_constant)

    def test(self):
        # print("SelectorDIC Test:")
        # print("self words:{}, length:{}".format(self.words, len(self.words)))

        # print("self hwords:{}, length:{}".format(self.hwords, len(self.hwords)))        
        print("This word:{}".format(self.this_word))
        val = self.hwords[self.this_word]
        print("This word val:{}, length of val[0] : {}, length of val [1] : {}".
            format(val,len(val[0]), len(val[1])))

        for key, val in self.hwords.items():
            if key == self.this_word:
                X, lengths = val
                print("This word val:{}, length of val[0] : {}, length of val [1] : {}".
                    format(val,len(val[0]), len(val[1])))
                print("array equals: X:{}, lengths:{}\nlen(X):{}, len(lengths):{}"
                    .format(X,lengths,len(X),len(lengths)))




class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        default_split = min(len(self.sequences),3)
 
        
        
        
        bestModel = None
        bestLogL = float('-inf')
        
        best_num_state = None
        avg_log_likelihood = {}
        # TODO implement model selection using CV
        for num_state in range(self.min_n_components,self.max_n_components+1):
            #hm_model = self.base_model(x)
            avg_logL = None
            scores = []
            
            if default_split >= 2:    
                split_method = KFold(n_splits = default_split)
                

                
                # hmm_model = None
                # logL = None 
                try:
                    
                    for cv_train_idx , cv_test_idx in split_method.split(self.sequences):
                        cv_train_X , cv_train_lengths = combine_sequences(cv_train_idx, self.sequences)
                        testX , testLengths = combine_sequences(cv_test_idx, self.sequences)
                        
                        hmm_model = GaussianHMM(n_components=num_state, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(cv_train_X,cv_train_lengths)



                        logL = hmm_model.score(testX, testLengths)

                        scores.append(logL)
                    avg_logL = np.mean(scores)

                except:
                    pass
            else:
                #default split < 2:
                try:
                    hmm_model = GaussianHMM(n_components=num_state, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(self.X,self.lengths)



                    logL = hmm_model.score(self.X,self.lengths)

                    scores.append(logL)
                    avg_logL = no.mean(scores)
                except:
                    pass


            #get average logL for current num_state
            # print("{} Average log likelihood :{} for num_state:{}".format(self.this_word,avg_logL,num_state))

            if not best_num_state and avg_logL is not None:
                bestLogL = avg_logL
                best_num_state = num_state
            if avg_logL is not None and avg_logL > bestLogL:
                bestLogL = avg_logL
                best_num_state = num_state
                # print("Bestmodel for {} now has {} states with logL of {}".format(self.this_word,num_state,avg_logL))
        
        if best_num_state:
            bestModel = GaussianHMM(n_components=best_num_state, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X,self.lengths)

    #                 print("Score: {}".format(score))
    #                 print("Train_Seq: {}".format(train_seq))
                
                
                #test_seq = [ self.sequences[j] for j in cv_test_idx ][0]

                
                
                
#                 print("Test_seq: {}".format(test_seq))
                
                
#                 hmm_model, logL = self.base(num_states, train_seq, test_seq)
#                 if not hmm_model:
#                     continue
        
        #print("BestModel return with state: {}".format())
        return bestModel if bestModel is not None else self.base_model(self.n_constant)
