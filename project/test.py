import requests

car = {
    "Levy": "500",
    "Manufacturer": "HYUNDAI",
    "Model": "Sonata",
    "Prod. year": "2015",
    "Category": "Sedan",
    "Leather interior": "No",
    "Fuel type": "Petrol",
    "Engine volume": "1.8",
    "Mileage": "2161 km",
    "Cylinders": 4.0,
    "Gear box type": "Automatic",
    "Drive wheels": "Front",
    "Doors": "04-May",
    "Wheel": "Left wheel",
    "Color": "Black",
    "Airbags": 12,
}

url = "http://localhost:9696/predict"
response = requests.post(url, json=car)
print(response.json())
