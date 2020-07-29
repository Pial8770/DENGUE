from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("dengue.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    f_class = model.predict(final)
    if f_class == [1]:
        if output > str(0.5):
            return render_template('dengue.html',pred='You have dengue.\nProbability of dengue is {}'.format(output),bhai="kuch karna hain iska ab?")
        else:
            return render_template('dengue.html',pred='You do not have dengue/chikungunya.\nProbability of dengue is {}'.format(output),bhai="kuch karna hain iska ab?")
    elif f_class == [2]:
        if output > str(0.5):
            return render_template('dengue.html',pred='You have chikungunya.\n Probability of chikungunya is {}'.format(output),bhai="Your Forest is Safe for now")
        else:
            return render_template('dengue.html',pred='You do not have chikungunya.\nProbability of chikungunya is {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('dengue.html',pred='You are safe.\n Probability of dengue/c is {}'.format(output),bhai="Your Forest is Safe for now")


if __name__ == '__main__':
    app.run(debug=True)
