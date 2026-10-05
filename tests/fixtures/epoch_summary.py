'''Small durable epoch-summary rows with every health value (P20).'''

def sample_health(epoch=0):
    return {
        'loss/task': 0.8 - 0.1 * epoch,
        'loss/code_code': 0.6 - 0.1 * epoch,
        'loss/radial': 1.2 - 0.1 * epoch,
        'loss/total': 2.6 - 0.3 * epoch,
        'logit_scale/task': 1.0 + 0.1 * epoch,
        'logit_scale/code_code': 1.2 + 0.1 * epoch,
        'radius/mean/level_2': 0.8 + 0.1 * epoch,
        'radius/sd/level_2': 0.1,
        'radius/mean/level_6': 4.0 + 0.1 * epoch,
        'radius/sd/level_6': 0.2
    }

def summary_rows():
    return [
        {
            'epoch': epoch,
            'mrr': 0.5 + 0.1 * epoch,
            **sample_health(epoch)
        } for epoch in range(3)
    ]
