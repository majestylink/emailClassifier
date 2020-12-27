from django.shortcuts import render


def home(request):
    return render(request, 'emailcls/index.html')


def getPredictions(emails):
    import pickle

    model = pickle.load(open("themodel.pickle", "rb"))
    scaled = pickle.load(open("thescaler.pickle", "rb"))

    prediction = model.predict(scaled.transform(emails))

    if prediction == 0:
        return "Email is not classified as spam"

    elif prediction == 1:
        return "Email is classified as spam"
    else:
        return "Can't classify email"


def result(request):
    emails = request.GET['emails']

    results = getPredictions([emails])
    return render(request, 'emailcls/result.html', {'result': results})
