from nueral  import * 

x_or_trainingdata = [
    ([0, 0], [0]),
    ([1, 0], [1]),
    ([0, 1], [1]),
    ([1, 1], [0])
]

xorn = NeuralNet(2, 20, 1)

xorn.train(x_or_trainingdata)

print()

print(xorn.test_with_expected(x_or_trainingdata))
# last 3 questions 
print("\n\nTraining voter opinion\n\n")

patient_data = [
    ([.9, .6, .8, .3, .1], [1]),
    ([.8, .8, .4, .6, .4], [1]),
    ([.7, .2, .4, .6, .3], [1]),
    ([.5, .5, .8, .4, .8], [0]),
    ([.3, .1, .6, .8, .8], [0]),
    ([.6, .3, .4, .3, .6], [0])
]

von = NeuralNet(5, 6, 1)

von.train(voter_opinion_data)

print(von.test_with_expected(voter_opinion_data))
#case data
test_data=[
    ([1,1,1,.1,.1]),
    ([.5,.2,.1,.7,.7]),
    ([.8,.3,.3,.3,.8]),
    ([.8,.3,.3,.8,.3]),
    ([.9,.8,.8,.3,.6]),
]
#cases Dem or rep
print(f"case 1: {test_data[0]} evaluates to: {von.evaluate(test_data[0])}")
print(f"case 2: {test_data[1]} evaluates to: {von.evaluate(test_data[1])}")
print(f"case 3: {test_data[2]} evaluates to: {von.evaluate(test_data[2])}")
print(f"case 4: {test_data[3]} evaluates to: {von.evaluate(test_data[3])}")
print(f"case 5: {test_data[4]} evaluates to: {von.evaluate(test_data[4])}")