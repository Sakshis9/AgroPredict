from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.contrib.auth import login as authlogin
from django.contrib.auth import logout as authlogout
from .models import Farmer, Wholesaler
from django.views.decorators.csrf import csrf_exempt

# Create your views here.
def index(request):
    if request.user.is_authenticated:
        utype = request.user.username
        utype = utype[utype.rfind('-') + 1:]
        context = {'utype': utype, 'username': request.user.username[:request.user.username.rfind('-')]}
        print(request.user.username, utype, "Ho raha hai")
    else:
        context = {'utype': None, 'username': None}
    return render(request, 'main.html', context)

def login(request):
    return render(request, 'login.html')

def signin(request):
    if request.method == 'POST':
        print("Signin Post request received.")
        password1 = request.POST['password']
        utype = request.POST['usertype']
        username = request.POST['username'] + "-" + utype
        print("***", username, password1, utype, "***")
        user = authenticate(username=username, password=password1)
        if user is not None:
            authlogin(request, user)
            print(user)
            return redirect('/')
        else:
            # Handle invalid login (optional)
            return render(request, 'login.html', {'error': 'Invalid credentials, please try again.'})

def signup(request):
    if request.method == 'POST':
        name = request.POST['fullname']
        phone = request.POST['phone number']
        password1 = request.POST['password']
        # password2 = request.POST['confirmpassword']
        utype = request.POST['usertype']
        username = request.POST['username'] + "-" + utype
        
        # Check if password and confirm password match
        # if password1 != password2:
        #     return render(request, 'signup.html', {'error': 'Passwords do not match, please try again.'})
        
        # Split the name into first and last name
        if ' ' in name:
            fname, lname = name.split(' ', 1)  # Split at the first space
        else:
            fname = name
            lname = ''  # Set lname to an empty string if no space is present

        print("***", name, username, phone, password1, utype, "***")

        # Check the user type and save the respective model
        if utype == 'farmer':
            farmer = Farmer(username=username, phone=phone, name=name)
            farmer.save()
        if utype == 'wholesaler':
            ws = Wholesaler(username=username, phone=phone, name=name)
            ws.save()

        # Creating the User
        user = User.objects.create_user(username=username, email=None, password=password1, first_name=fname, last_name=lname)
        return redirect('/login')

def logout(request):
    if request.method == 'POST':
        authlogout(request)
        return redirect('/login')
    return redirect('/login')
