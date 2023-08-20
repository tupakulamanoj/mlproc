from flask import Flask,request,render_template
from src.pipeline.predicting_pipeline import predict_data,predict
from src.exception import custom_exception
import sys

app=Flask(__name__)


@app.route('/',methods=['GET',"POST"])
def prediction():
    try:
        
        if request.method =='GET':
            return render_template('index.html')
        else:
            data=predict_data(gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('writing_score')),
                writing_score=float(request.form.get('reading_score'))
            )
        data_frame=data.data_as_dataframe()
        prediction=predict()
        res=prediction.predictdata(data_frame)
        
        return render_template('index.html',results=res[0])
    except Exception as e:
        raise custom_exception(e,sys)
    
if __name__=='__main__':
    app.run(debug=True)
        
    
    
    