from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import os
import numpy as np

model = VGG16(weights='imagenet')

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)

            result = model.predict(img_array)
            prediction = decode_predictions(result, top=5)[0] 
            
            total_score = sum([score for _, _, score in prediction])
            prediction_with_percentage = [(label, description, score / total_score * 100) for label, description, score in prediction]
            print(prediction_with_percentage)

            img_data = request.POST.get('img_data')
            return render(request, 'home.html', {
                'form': form,
                'prediction': prediction_with_percentage, 
                'img_data': img_data
            })
        
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})
