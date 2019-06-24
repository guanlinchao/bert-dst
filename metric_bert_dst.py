import glob
import json
import sys
import numpy as np


def get_joint_slot_correctness(fp,
                               key_class_label_id='class_label_id',
                               key_class_prediction='class_prediction',
                               key_start_pos='start_pos',
                               key_start_prediction='start_prediction',
                               key_end_pos='end_pos',
                               key_end_prediction='end_prediction'):
  with open(fp) as f:
    preds = json.load(f)
    class_correctness = []
    pos_correctness = []
    total_correctness = []

    for pred in preds:
      guid = pred['guid']
      turn_gt_class = pred[key_class_label_id]
      turn_pd_class = pred[key_class_prediction]
      gt_start_pos = pred[key_start_pos]
      pd_start_pos = pred[key_start_prediction]
      gt_end_pos = pred[key_end_pos]
      pd_end_pos = pred[key_end_prediction]

      if guid.split('-')[-1] == '0': # First turn, reset the slots
        joint_gt_class = turn_gt_class
        joint_gt_start_pos = gt_start_pos
        joint_gt_end_pos = gt_end_pos
        joint_pd_class = turn_pd_class
        joint_pd_start_pos = pd_start_pos
        joint_pd_end_pos = pd_end_pos
      else:
        if turn_gt_class > 0:
          joint_gt_class = turn_gt_class
          joint_gt_start_pos = gt_start_pos
          joint_gt_end_pos = gt_end_pos
        if turn_pd_class > 0:
          joint_pd_class = turn_pd_class
          joint_pd_start_pos = pd_start_pos
          joint_pd_end_pos = pd_end_pos

      total_correct = True
      if joint_gt_class == joint_pd_class:
        class_correctness.append(1.0)
        if joint_gt_class == 2:
          #overlap = set(range(joint_gt_start_pos, joint_gt_end_pos+1)).intersection(range(joint_pd_start_pos, joint_pd_end_pos+1))

          if joint_gt_start_pos == joint_pd_start_pos and joint_gt_end_pos == joint_pd_end_pos:
          # if joint_gt_start_pos <= joint_pd_start_pos and joint_gt_end_pos >= joint_pd_end_pos:
          # if joint_gt_start_pos <= joint_pd_end_pos or joint_gt_end_pos >= joint_pd_start_pos:
          # if len(overlap) > 0:
            pos_correctness.append(1.0)
          else:
            #print('Wrong position prediciton: guid %s, gt=(%d, %d), pd=(%d, %d)' % (guid, joint_gt_start_pos, joint_gt_end_pos, joint_pd_start_pos, joint_pd_end_pos))
            pos_correctness.append(0.0)
            total_correct = False
      else:
        class_correctness.append(0.0)
        total_correct = False
      if total_correct:
        total_correctness.append(1.0)
      else:
        total_correctness.append(0.0)

    return np.asarray(total_correctness), np.asarray(class_correctness), np.asarray(pos_correctness)


if __name__ == "__main__":
  acc_list = []
  key_class_label_id = 'class_label_id_%s'
  key_class_prediction = 'class_prediction_%s'
  key_start_pos = 'start_pos_%s'
  key_start_prediction = 'start_prediction_%s'
  key_end_pos = 'end_pos_%s'
  key_end_prediction = 'end_prediction_%s'


  for fp in sorted(glob.glob(sys.argv[2])):
    print(fp)
    goal_correctness = 1.0
    dataset = sys.argv[1].lower()
    if dataset in ['woz2', 'dstc2_clean']:
      slots = ['area', 'food', 'price range']
    elif dataset == 'sim-m':
      slots = ['date', 'movie', 'time', 'num_tickets', 'theatre_name']
    elif dataset == 'sim-r':
      slots = ['category', 'rating', 'num_people', 'location', 'restaurant_name',
       'time', 'date', 'price_range', 'meal']
    for slot in slots:
      tot_cor, cls_cor, pos_cor = get_joint_slot_correctness(fp,
                                                             key_class_label_id=(key_class_label_id % slot),
                                                             key_class_prediction=(key_class_prediction % slot),
                                                             key_start_pos=(key_start_pos % slot),
                                                             key_start_prediction=(key_start_prediction % slot),
                                                             key_end_pos=(key_end_pos % slot),
                                                             key_end_prediction=(key_end_prediction % slot)
                                                             )
      print('%s: joint slot acc: %g, class acc: %g, position acc: %g' % (slot, np.mean(tot_cor), np.mean(cls_cor), np.mean(pos_cor)))
      goal_correctness *= tot_cor

    acc = np.mean(goal_correctness)
    acc_list.append((fp, acc))
  acc_list_s = sorted(acc_list, key=lambda tup: tup[1], reverse=True)
  for (fp, acc) in acc_list_s:
    print('Joint goal acc: %g, %s' % (acc, fp))