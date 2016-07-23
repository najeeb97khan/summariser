from django.shortcuts import render
from Summarisation import *
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
	# print request.POST
	data = request.POST.get('data',False)
	if data:
		data = data.strip()
		a = Document(data)
		a.construct_graph()
		x = textrank_weighted(a)
		temp = []
		for i in x[0:5]:
			temp.append(i[0])
		context = {"data":temp}
		template = "result.html"
	else:
		context = {}
		template = "text.html"
	return render(request,template,context)