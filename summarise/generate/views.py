from django.shortcuts import render

# Create your views here.

def home(request):
    context = {}
    template = "home.html"
    return render(request,template,context)

def products(request):
	context = {}
	template = "products.html"
	return render(request,template,context)
def text(request):
	print request.POST
	context = {}
	template = "text.html"
	return render(request,template,context)