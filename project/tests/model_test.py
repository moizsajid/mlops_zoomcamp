import model

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


def test_predict():
    model_service = model.init()

    actual_prediction = model_service.predict(car)
    expected_prediction = 45216.45703125

    assert actual_prediction == expected_prediction


def test_return_type():
    model_service = model.init()

    actual_prediction = model_service.predict(car)

    assert isinstance(actual_prediction, float)
