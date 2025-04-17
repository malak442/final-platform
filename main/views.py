

from django.shortcuts import render

def index(request):
    return render(request, 'main/index.html')

def about(request):
    return render(request, 'main/about.html')

def contact(request):
    return render(request, 'main/contact.html')

def project(request):
    return render(request, 'main/login.html')
def project(request):
    return render(request, 'main/signup.html')
def service(request):
    return render(request, 'main/service.html')

def team(request):
    return render(request, 'main/team.html')

def custom_404(request, exception=None):
    return render(request, 'main/404.html', status=404)

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib import messages
from .forms import ContactForm

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('chat_with_ai')  # ✅ send to AI chat page
    else:
        form = AuthenticationForm()
    return render(request, 'main/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'main/signup.html', {'form': form})

def contact_view(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Your message was successfully sent.")
            return redirect('contact')
    else:
        form = ContactForm()
    return render(request, 'main/contact.html', {'form': form})




import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .ai_agent import generate_caption_from_xray

def chat_with_ai(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('xray_image'):
        image = request.FILES['xray_image']
        fs = FileSystemStorage()
        # Save uploaded file into 'media/' folder
        image_name = fs.save(image.name, image)
        image_path = fs.path(image_name)

        result = generate_caption_from_xray(image_path)
        context = {
            'user_message': "Sent an X-ray image.",
            'ai_response': result["caption"],
            # Display the gradcam image by prepending MEDIA_URL (e.g., /media/gradcam_output.png)
            'gradcam_path': settings.MEDIA_URL + result["gradcam_path"],
        }
    return render(request, 'main/chat_with_ai.html', context)


