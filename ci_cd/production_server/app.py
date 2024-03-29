from workerA import add_nums, get_accuracy, get_predictions

from flask import (
   Flask,
   request,
   jsonify,
   Markup,
   render_template 
)

#app = Flask(__name__, template_folder='./templates',static_folder='./static')
app = Flask(__name__)

@app.route("/")
def index():
    return '<h1>Welcome to the Machine Learning Course.</h1>'

@app.route("/accuracy", methods=['POST', 'GET'])
def accuracy():
    if request.method == 'POST':
        r = get_accuracy.delay()
        a = r.get()
        return '<h1>The accuracy is {}</h1>'.format(a)

    return '''<form method="POST">
    <input type="submit">
    </form>'''

@app.route("/predictions", methods=['POST', 'GET'])
def predictions():
    if request.method == 'POST':
        final_results = get_predictions()
        #predictions = results.get()
        
        accuracy = get_accuracy(final_results["predicted"])
        #accuracy = results.get()
      
        #print(final_results['y'])
        return render_template('result.html', accuracy=accuracy ,final_results=final_results) 
                    
    return '''<form method="POST">
    <input type="submit">
    </form>'''

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=5100,debug=True)
