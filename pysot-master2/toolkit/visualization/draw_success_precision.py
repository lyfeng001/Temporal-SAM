import matplotlib.pyplot as plt
import numpy as np

from .draw_utils import COLOR, LINE_STYLE

def draw_success_precision(success_ret, name, videos, attr, precision_ret=None,
        norm_precision_ret=None, bold_name=None, axis=[0, 1]):
    # success plot
    # plt.figure(figsize=(60, 5), dpi=400)
    fig, ax = plt.subplots(dpi=400)
    ax.grid(b=True)
    ax.set_aspect(1)
    plt.xlabel('Overlap threshold',fontsize=14,fontweight='bold')
    plt.ylabel('Success rate',fontsize=14,fontweight='bold')
    if attr == 'ALL':
        plt.title(r'\textbf{Success plots on %s}' % (name),fontsize=14)
    else:
        plt.title(r'\textbf{Success plots - %s}' % (attr))
    plt.axis([0, 1]+axis)
    success = {}
    thresholds = np.arange(0, 1.05, 0.05)
    for tracker_name in success_ret.keys():
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        success[tracker_name] = np.mean(value)
    for idx, (tracker_name, auc) in  \
            enumerate(sorted(success.items(), key=lambda x:x[1], reverse=True)):
        if tracker_name == bold_name:
            label = r"\textbf{[%.3f] %s}" % (auc, tracker_name)
        else:
            label = "[%.3f] " % (auc) + tracker_name
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        plt.plot(thresholds, np.mean(value, axis=0),
                color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=1)
    ax.legend(loc='lower left', labelspacing=0.2)
    ax.autoscale(enable=True, axis='both', tight=True)
    xmin, xmax, ymin, ymax = plt.axis()
    ax.autoscale(enable=False)
    ymax += 0.01#0.03
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xticks(np.arange(xmin, xmax+0.01, 0.1))
    plt.yticks(np.arange(ymin, ymax, 0.1))
    ax.set_aspect((xmax - xmin)/(ymax-ymin))

    a = np.arange(0, 1.1, 0.2)
    b = np.arange(0, 0.72, 0.2)
    plt.xticks(a)
    plt.yticks(b)
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))

    plt.savefig('J:/study/kd_lowillum/pysot-master2/tools/output/success.png', dpi=900,
                bbox_inches='tight')
    plt.show()

    if precision_ret:
        # norm precision plot
        fig, ax = plt.subplots(dpi=400)
        ax.grid(b=True)
        ax.set_aspect(50)

        plt.xlabel('Location error threshold',fontsize=14,fontweight='bold')
        plt.ylabel('Precision',fontsize=14,fontweight='bold')
        if attr == 'ALL':
            plt.title(r'\textbf{Precision plots on %s}' % (name),fontsize=14)
        else:
            plt.title(r'\textbf{Precision plots - %s}' % (attr))
        plt.axis([0, 50]+axis)
        precision = {}
        thresholds = np.arange(0, 51, 1)
        for tracker_name in precision_ret.keys():
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=1)
        ax.legend(loc='lower right', labelspacing=0.2)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.01#0.03
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 5))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        # plt.legend(loc='lower left',bbox_to_anchor=(1.1,0))
        a = np.arange(0, 51, 10)
        b = np.arange(0, 0.72, 0.2)
        plt.xticks(a)
        plt.yticks(b)
        plt.rcParams['figure.figsize'] = (10, 5)
        plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))
        plt.savefig('J:/study/kd_lowillum/pysot-master2/tools/output/precision.eps',
                    bbox_inches='tight')
        plt.show()
        # plt.savefig('/media/lyf/工作/study/kd_lowillum/pysot-master2/toolkit/visualization/outcome/precision.png',
        #             dpi=900,
        #             bbox_inches='tight')
    # norm precision plot
    if norm_precision_ret:
        fig, ax = plt.subplots()
        ax.grid(b=True)
        plt.xlabel('Location error threshold')
        plt.ylabel('Precision')
        if attr == 'ALL':
            plt.title(r'\textbf{Normalized Precision plots of OPE on %s}' % (name))
        else:
            plt.title(r'\textbf{Normalized Precision plots of OPE - %s}' % (attr))
        norm_precision = {}
        thresholds = np.arange(0, 51, 1) / 100
        for tracker_name in precision_ret.keys():
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            norm_precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(norm_precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
        ax.legend(loc='lower right', labelspacing=0.2)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 0.05))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        plt.show()
