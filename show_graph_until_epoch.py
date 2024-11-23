import matplotlib.pyplot as plt
from IPython.display import clear_output

# 원본: https://github.com/seonydg/LSTM-for-Anomaly-Detection

def record_logs(logs, epoch, tloss, tacc, vloss, vacc, time_spent):
    # Train Log 기록
    logs['epoch'].append(epoch)
    logs['tloss'].append(tloss)
    logs['tacc'].append(tacc)
    logs['vloss'].append(vloss)
    logs['vacc'].append(vacc)
    logs['time'].append(time_spent)
    return logs


def print_log(logs, epoch, num_epochs, log_stack, no_improve, best_loss):
    # 학습 추이 출력 : 소숫점 3자리까지
    train_loss = round(float(logs['tloss'][-1]), 3)
    train_acc = round(float(logs['tacc'][-1]), 3)
    val_loss = round(float(logs['vloss'][-1]), 3)
    val_acc = round(float(logs['vacc'][-1]), 3)
    time_spent = round(float(logs['time'][-1]), 3)

    log_str = f'Epoch: {epoch+1}/{num_epochs} | T_Loss {train_loss:5} | T_Acc {train_acc:5} | V_Loss {val_loss:5} | V_Acc {val_acc:5} | Time {time_spent:5}'
    if no_improve == 0:
        log_str += f" | Best model saved: {best_loss:.4f}"
    log_stack.append(log_str)
    
    # 학습 추이 그래프 출력
    hist_fig, loss_axis = plt.subplots(figsize=(10, 3), dpi=99)
    hist_fig.patch.set_facecolor('white')

    # Loss Line 구성
    loss_t_line = plt.plot(logs['epoch'], logs['tloss'], label='Train_Loss', color='red', marker='o')
    loss_v_line = plt.plot(logs['epoch'], logs['vloss'], label='Valid_Loss', color='blue', marker='s')
    loss_axis.set_xlabel('epoch')
    loss_axis.set_ylabel('loss')

    # Acc, Line 구성
    acc_axis = loss_axis.twinx()
    acc_t_line = acc_axis.plot(logs['epoch'], logs['tacc'], label='Train_Acc', color='red', marker='+')
    acc_v_line = acc_axis.plot(logs['epoch'], logs['vacc'], label='Valid_Acc', color='blue', marker='x')
    acc_axis.set_ylabel('accuracy')

    # 그래프 출력
    hist_lines = loss_t_line + loss_v_line + acc_t_line + acc_v_line
    loss_axis.legend(hist_lines, [l.get_label() for l in hist_lines])
    loss_axis.grid()
    plt.title(f'Learning history until epoch {epoch}')
    plt.draw()

    # 텍스트 로그 출력
    clear_output(wait=True)
    plt.show()
    for idx in reversed(range(len(log_stack))):
        print(log_stack[idx])
    
    return log_stack