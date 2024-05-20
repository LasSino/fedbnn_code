import torch

def accuracy(test_dataset,model,monte_carlo_times=1,cuda=False):
    accuracies=[]
    with torch.no_grad():
        for batch_data in test_dataset:
            test_x,test_y=batch_data
            if cuda==True:
                test_x=test_x.to(torch.device('cuda'))
                test_y=test_y.to(torch.device('cuda'))
            cumulated_prediction=None
            for _ in range(monte_carlo_times):
                prediction=model(test_x)
                if cumulated_prediction is None:
                    cumulated_prediction=torch.zeros_like(prediction)
                cumulated_prediction+=prediction
            
            _, predicted = torch.max(cumulated_prediction.data, 1)
            total = test_y.size(0)
            correct = (predicted == test_y).sum()
            accuracies.append(correct/total)

    return (sum(accuracies)/len(accuracies)).item()
