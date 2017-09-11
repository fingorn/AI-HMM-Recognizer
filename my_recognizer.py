import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """


    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # probabilities = list(dict(None))* len(test_set.get_all_sequences())

    probabilities = [ {} for x in range(len(test_set.get_all_sequences()))]
    # print("Probabilities: {}, len:{}".format(probabilities,len(probabilities)))
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    print("in recognize")
    # print("Models dic:\n{}\n len(models):{}".format(models,len(models)))

    # print("test_set singlesData:{}\nget_all_sequences :{}\n len(get_all_sequences):{}".format(test_set,test_set.get_all_sequences(),len(test_set.get_all_sequences())))
    # print("test_set get_all_Xlengths:{} len:{} \n".format( test_set.get_all_Xlengths(),len(test_set.get_all_Xlengths())))
    # print("test_set get_all_Xlengths()[177]:\n{}".format(test_set.get_all_Xlengths()[177]))    


    for key, value in test_set.get_all_sequences().items():

      # print("key:{}, value:{}".format(key,value))
      # print("get_all_Xlengths()[{}]: {}".format(key,test_set.get_all_Xlengths()[key]))
      X, lenghts = test_set.get_all_Xlengths()[key] 
      prob = {}

      best_logL = float('-inf')
      best_word = None
      for word,model in models.items():
        try:

          score = model.score(X,lenghts) 
          # word_score = {}
          # word_score[word] = score
          probabilities[key][word] =  score

          if best_word == None:
            best_logL = score
            best_word = word

          if score > best_logL:
            best_logL = score
            best_word = word
        except Exception as e:
          pass

      #update guesses with best score
      guesses.append(best_word)


    # print("Probabilities final : \n{},\nlen:{}".format(probabilities,len(probabilities)))
    # print("Guesses final:\n{},\nlen:{}".format(guesses,len(guesses)))

    return probabilities, guesses
