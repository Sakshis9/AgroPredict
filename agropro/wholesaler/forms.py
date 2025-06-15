# wholesaler/forms.py
from django import forms
from .models import Wholesaler  # Assuming you want to edit the Wholesaler model

class ProfileForm(forms.ModelForm):
    class Meta:
        model = Wholesaler  # You might want to use the Wholesaler model or another model
        fields = ['name', 'state', 'email', 'address', 'phone', 'profile_picture']  # Add any fields from your model
