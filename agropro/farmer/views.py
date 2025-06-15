import os
import cv2
import json
import numpy as np
import pandas as pd
from django.apps import apps
from django.conf import settings
from django.http import HttpResponse
from django.http import JsonResponse
from .extras.svm import CropRecommenderSVM
from django.shortcuts import render, redirect
from .extras.soil_classifier import SoilClassifier
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required


BASE_DIR = settings.BASE_DIR

svm_model_path = os.path.join(BASE_DIR,'farmer', 'extras', 'svm_crop_model.pkl')
label_encoder_path = os.path.join(BASE_DIR, 'farmer', 'extras', 'label_encoder.pkl')


# /Users/sakshi/Downloads/AgroPro_Final/agropro/farmer/uploads/CropRecommendation.csv
csv_path = os.path.join(settings.BASE_DIR.parent, 'agropro', 'farmer', 'uploads', 'CropRecommendation.csv')
SVM_crop_recommender = CropRecommenderSVM(csv_path=csv_path, train_model=False)

 
# SVM_crop_recommender = CropRecommenderSVM(csv_path=None, train_model=False)
SVM_crop_recommender.load_model(svm_model_path, label_encoder_path)


MODEL_PATH = os.path.join(settings.BASE_DIR, 'farmer/extras/soil_model.h5')
DATA_DIR = os.path.join(settings.BASE_DIR, 'farmer/extras/Soil_Types')
UNSPLASH_ACCESS_KEY = "XQDQPal0uOTMihgN21Dol03gpWZAyAD_Cw294_I9M1c"
CNN_obj_3D = SoilClassifier(DATA_DIR, model_path=MODEL_PATH)

Farmer = apps.get_model('home', 'Farmer')
Wholesaler = apps.get_model('home', 'Wholesaler')
Crop = apps.get_model('home', 'Crop')
Notification = apps.get_model('home', 'Notification')

base_dir = os.path.dirname(__file__)
print(base_dir)

df = pd.read_csv(base_dir+'/uploads/crop_price_dataset.csv')
df1 = pd.read_csv(base_dir+'/uploads/Data Yield_hect - Sheet1.csv')
print(df[:3])
print(df1[:3])

dfReq = pd.read_csv(base_dir+'/uploads/CropRecommendationNEWCSV2.csv')
# Clean up strings
for col in ['States', 'Crops for mixed cropping', 'Soil type', 'Season']:
    dfReq[col] = dfReq[col].astype(str).str.strip()

# Convert number-like columns to float
dfReq['Expected revenues'] = pd.to_numeric(dfReq['Expected revenues'], errors='coerce')
dfReq['Quantity of per hectare'] = pd.to_numeric(dfReq['Quantity of per hectare'], errors='coerce')

def findPrice(data):
    state = data['state'].strip()
    crop = data['crop'].strip()

    newPd = dfReq.loc[
        (dfReq['States'].str.strip().str.lower() == state.lower()) &
        (dfReq['Crops for mixed cropping'].str.lower().str.contains(crop.lower()))
    ]

    if newPd.empty:
        print(f"[Price] No match for: State = '{state}', Crop = '{crop}'")
        return 0

    try:
        expected_revenue = float(newPd['Expected revenues'].iloc[0])
        return expected_revenue * float(data['production'])
    except Exception as e:
        print(f"[Price] Error for crop={crop}: {e}")
        return 0

def findYield(data):
    state = data['state'].strip()
    crop = data['crop'].strip()

    newPd = dfReq.loc[
        (dfReq['States'].str.strip().str.lower() == state.lower()) &
        (dfReq['Crops for mixed cropping'].str.lower().str.contains(crop.lower()))
    ]

    if newPd.empty:
        print(f"[Yield] No match for: State = '{state}', Crop = '{crop}'")
        return 0

    try:
        quantity = float(newPd['Quantity of per hectare'].iloc[0])
        return quantity * float(data['area'])
    except Exception as e:
        print(f"[Yield] Error for crop={crop}: {e}")
        return 0

newData = {
    'area':1,
    'crop':'Soyabean',
    'state':'Maharashtra',
    'season':'Kharif'    
}

print("Checking for Peas and Beans:",findYield(newData))

def prediction(request):
    if request.user.is_authenticated:
        username = request.user.username
        username = username[:username.rfind('-')]
        print(username, "Ho raha hai")

        instance = Farmer.objects.filter(username=request.user.username).values()[0]

        if request.method == 'POST':
            print('Request received')
            season = request.POST.get('season')
            print('Season:', season)
            print('Image File:', request.FILES['file'])

            imageFile = request.FILES['file']

            # Read image file to OpenCV format
            image_bytes = np.frombuffer(imageFile.read(), np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

            if image is None:
                return JsonResponse({'error': 'Invalid image format'}, status=400)

            # Predict soil type
            soil_type = CNN_obj_3D.predict_soil_type(image)
            print("Predicted soil type:", soil_type)

            # Recommend crops using SVM
            recommended_crops = SVM_crop_recommender.recommend_crops(
                soil_type,
                instance['state'],
                season
            )
            print("Recommended crops:", recommended_crops)

            finalArray = []
            for crop_string in recommended_crops:
                for crop in crop_string.split(','):
                    crop = crop.strip()
                    data = {
                        'area': instance['area'],
                        'crop': crop,
                        'state': instance['state'],
                        'season': season
                    }

                    production = findYield(data)
                    data['production'] = production

                    priceAssumption = findPrice(data)
                    data['price'] = priceAssumption

                    finalArray.append(data)

            print("Final output:", finalArray)
            return JsonResponse({'result': finalArray})

        elif not all([instance.get('state'), instance.get('area'), instance.get('email'), instance.get('address')]):
            print('Incomplete profile:', instance)
            return redirect('/farmer/profile')

        else:
            return render(request, 'predic.html')

    else:
        return redirect('/login')

@login_required(login_url='login')
def predict_soil(request):
    username = request.user.username
    trimmed_username = username[:username.rfind('-')]
    print(trimmed_username, "Ho raha hai")

    instance = Farmer.objects.filter(username=username).first()
    if not instance:
        return JsonResponse({'error': 'Farmer not found'}, status=404)

    if request.method == 'POST':
        print('Request received')

        # Get the season and uploaded file
        season = request.POST.get('season')
        image_file = request.FILES.get('file')

        print('Season:', season)
        print('Image File:', image_file)

        if not image_file:
            return JsonResponse({'error': 'No image uploaded'}, status=400)

        try:
            # Convert to OpenCV image
            image_bytes = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Image decoding failed")

            # Predict soil type
            soil_type = CNN_obj_3D.predict_soil_type(image)
            print("Predicted soil type:", soil_type)

            return JsonResponse({'soil_type': soil_type})
        
        except Exception as e:
            print("Error during prediction:", str(e))
            return JsonResponse({'error': 'Failed to process image or predict soil'}, status=500)

    return render(request, 'soilType.html')



def notification(request):
    if request.user.is_authenticated:
        utype=request.user.username
        utype=utype[utype.rfind('-')+1:]

        if request.method=='POST':
            not_id=request.POST['id']
            notif_obj=list(Notification.objects.filter(id=not_id))[0]
            notif_obj.accepted=True
            notif_obj.save()

        f_obj=list(Farmer.objects.filter(username=request.user.username))[0]
        notif_final=[]
        notif_temp=list(Notification.objects.all())
        for i in notif_temp:
            if i.crop.farmer.id == f_obj.id:
                notif_final.append([])
                notif_final[-1].append(i.crop.name)
                notif_final[-1].append(i.wholesaler.name)
                notif_final[-1].append(i.crop.price)
                notif_final[-1].append(i.accepted)
                notif_final[-1].append("91"+i.wholesaler.phone)
                notif_final[-1].append(i.id)
        print(notif_final)

        # Order By
        ob=request.GET.get('orderby')
        if ob=="Pending":
            crops=sorted(notif_final, key=lambda x: x[3])
        if ob=="Accepted":
            crops=(sorted(notif_final, key=lambda x: x[3])).reverse()

        context={'utype':utype,'username':request.user.username[:request.user.username.rfind('-')],"notif":notif_final,"n":3}
        return render(request,'notif.html',context)
    else:
        return redirect('/login')
    

def profile(request):
    if request.user.is_authenticated:
        username=request.user.username
        username=username[:username.rfind('-')]
        # context={'utype':utype,'username':request.user.username[:request.user.username.rfind('-')+1]}
        print(username,"Ho raha hai")
        instance = Farmer.objects.filter(username = request.user.username).values()[0]
        print("Checking for instance: ",instance)
        crops = Crop.objects.filter(farmer = instance['id'],available = True).values()
        
        print("The instance is:",instance)
        print("The crop instance  is:",crops)
        if request.GET.get('newCrop'):
            newCrop = True
        else:
            newCrop = False

        if request.GET.get('itemDel'):
            itemDel = True
        else:
            itemDel = False

        context = {
            'instance':instance,
            'crops':crops,
            'newCrop':newCrop,
            'itemDel':itemDel
        }
        print("Checkinf for variables:",context['newCrop'],context['itemDel'])
        return render(request,'profile.html',context)
    else:
        return redirect('/')


def editProfile(request):
    if request.method == 'POST' and request.user.is_authenticated:
        username=request.user.username
        username=username[:username.rfind('-')]
        # context={'utype':utype,'username':request.user.username[:request.user.username.rfind('-')+1]}
        print(username,"Ho raha hai")
        instance = Farmer.objects.filter(username = request.user.username).update(name = request.POST['name'],state = request.POST['state'],area = request.POST['area'],email = request.POST['email'],address = request.POST['address'],phone = request.POST['phone'])
        
        print("The instance is:",instance)
        print("The new changes are:",request.POST['state'],request.POST['name'],request.POST['phone'],request.POST['area'],request.POST['email'],request.POST['address'])

        return redirect('/farmer/profile')

def setCrop(request):
    if request.method == 'POST':
        print(request.POST['name'],request.POST['quantity'],request.POST['price'],request.POST['tsp'])
        
        username=request.user.username
        username=username[:username.rfind('-')]
        # context={'utype':utype,'username':request.user.username[:request.user.username.rfind('-')+1]}
        print(username,"Ho raha hai")
        instance = Farmer.objects.filter(username = request.user.username).values()[0]
        
        print("The instance is:",instance)
        
        newCrop = Crop.objects.create(farmer=Farmer(id=instance['id']), name = request.POST['name'],quantity = request.POST['quantity'],price = request.POST['price'],available = True)
        print('The new crop is:',newCrop)

        return redirect('/farmer/profile?newCrop=True')

def get_crop_image(crop_name):
    """
    Fetch a relevant crop image from Unsplash API.
    If an image is not found, return a placeholder image.
    """
    url = f"https://api.unsplash.com/photos/random?query={crop_name}&client_id={UNSPLASH_ACCESS_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["urls"]["regular"]  # Return Unsplash image URL
    except Exception as e:
        print(f"Error fetching image for {crop_name}: {e}")

    return "/Users/sakshi/Downloads/AgroPro_Final/agropro/static/images/premium_photo-1663945778994-11b3201882a0.avif"  # Default image if API fails

def removeCrop(request,crop_id):
    if request.method == "POST":
        print("Yes reached here:",crop_id)
        instance = Crop.objects.filter(id = crop_id).delete()
        print("Checking for deletion:",instance)
        return redirect('/farmer/profile?itemDel=True')
    else:
        return redirect('/farmer/profile')

def blog_list(request):
    return render(request, 'blog_detail.html') 