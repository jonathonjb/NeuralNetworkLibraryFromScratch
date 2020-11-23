class Accuracy:
    def evaluate(self, predictions, y):
        numCorrect = 0
        for prediction, y_curr in zip(predictions[0], y):
            if (prediction == y_curr):
                numCorrect += 1
        accuracy = numCorrect / y.shape[0]
        self.accuracy = accuracy
        return accuracy

class Recall:
    def evaluate(self, predictions, y):
        numTruePositives = 0
        numTruePositivesLabeledPositive = 0

        for prediction, y_curr in zip(predictions[0], y):
            if(y_curr == 1):
                numTruePositives += 1
                if(prediction == 1):
                    numTruePositivesLabeledPositive += 1
        recall = numTruePositivesLabeledPositive / numTruePositives
        self.recall = recall
        return recall

class Precision:
    def evaluate(self, predictions, y):
        numLabeledPositive = 0
        numLabeledPositiveTruePositive = 0

        for prediction, y_curr in zip(predictions[0], y):
            if(prediction == 1):
                numLabeledPositive += 1
                if(y_curr == 1):
                    numLabeledPositiveTruePositive += 1

        precision = numLabeledPositiveTruePositive / numLabeledPositive
        self.precison = precision
        return precision