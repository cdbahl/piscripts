from sense_hat import SenseHat
sense = SenseHat()
#sense.show_message("I love Jessica!")
sense.clear()

pressure = sense.get_pressure()
print(f"pressure {pressure}")

tempp = sense.get_temperature_from_pressure()
print(f"temp from pressure {tempp}")

temph = sense.get_temperature_from_humidity()
print(f"temp from humidity {temph}")

humidity = sense.get_humidity()
print(f"humidity {humidity}")

