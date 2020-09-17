import json
import re
import sys

import tensorflow as tf

import util

# Directory of bert, cloned from github.com/google-research/bert
sys.path.append("/path/to/bert")
import tokenization


SEMANTIC_DICT = {
  'center': ['centre', 'downtown', 'central', 'down town', 'middle'],
  'south': ['southern', 'southside'],
  'north': ['northern', 'uptown', 'northside'],
  'west': ['western', 'westside'],
  'east': ['eastern', 'eastside'],
  'east side': ['eastern', 'eastside'],

  'cheap': ['low price', 'inexpensive', 'cheaper', 'low priced', 'affordable',
            'nothing too expensive', 'without costing a fortune', 'cheapest',
            'good deals', 'low prices', 'afford', 'on a budget', 'fair prices',
            'less expensive', 'cheapeast', 'not cost an arm and a leg'],
  'moderate': ['moderately', 'medium priced', 'medium price', 'fair price',
               'fair prices', 'reasonable', 'reasonably priced', 'mid price',
               'fairly priced', 'not outrageous','not too expensive',
               'on a budget', 'mid range', 'reasonable priced', 'less expensive',
               'not too pricey', 'nothing too expensive', 'nothing cheap',
               'not overpriced', 'medium', 'inexpensive'],
  'expensive': ['high priced', 'high end', 'high class', 'high quality',
                'fancy', 'upscale', 'nice', 'fine dining', 'expensively priced'],

  'afghan': ['afghanistan'],
  'african': ['africa'],
  'asian oriental': ['asian', 'oriental'],
  'australasian': ['australian asian', 'austral asian'],
  'australian': ['aussie'],
  'barbeque': ['barbecue', 'bbq'],
  'basque': ['bask'],
  'belgian': ['belgium'],
  'british': ['cotto'],
  'canapes': ['canopy', 'canape', 'canap'],
  'catalan': ['catalonian'],
  'corsican': ['corsica'],
  'crossover': ['cross over', 'over'],
  'gastropub': ['gastro pub', 'gastro', 'gastropubs'],
  'hungarian': ['goulash'],
  'indian': ['india', 'indians', 'nirala'],
  'international': ['all types of food'],
  'italian': ['prezzo'],
  'jamaican': ['jamaica'],
  'japanese': ['sushi', 'beni hana'],
  'korean': ['korea'],
  'lebanese': ['lebanse'],
  'north american': ['american', 'hamburger'],
  'portuguese': ['portugese'],
  'seafood': ['sea food', 'shellfish', 'fish'],
  'singaporean': ['singapore'],
  'steakhouse': ['steak house', 'steak'],
  'thai': ['thailand', 'bangkok'],
  'traditional': ['old fashioned', 'plain'],
  'turkish': ['turkey'],
  'unusual': ['unique and strange'],
  'venetian': ['vanessa'],
  'vietnamese': ['vietnam', 'thanh binh'],
                  }

FIX = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}


def get_token_pos(tok_list, label):
  find_pos = []
  found = False
  label_list  = [item for item in map(str.strip, re.split("(\W+)", label)) if len(item) > 0]
  len_label = len(label_list)
  for i in range(len(tok_list) + 1 - len_label):
    if tok_list[i:i+len_label] == label_list:
      find_pos.append((i,i+len_label))  # start, exclusive_end
      found = True
  return found, find_pos


def check_label_existence(label, usr_utt_tok, sys_utt_tok):
  in_usr, usr_pos = get_token_pos(usr_utt_tok, label)
  in_sys, sys_pos = get_token_pos(sys_utt_tok, label)

  if not in_usr and not in_sys and label in SEMANTIC_DICT:
    for tmp_label in SEMANTIC_DICT[label]:
      in_usr, usr_pos = get_token_pos(usr_utt_tok, tmp_label)
      in_sys, sys_pos = get_token_pos(sys_utt_tok, tmp_label)
      if in_usr or in_sys:
        label = tmp_label
        break
  return label, in_usr, usr_pos, in_sys, sys_pos


def get_turn_label(label, sys_utt_tok, usr_utt_tok, slot_last_occurrence):
  sys_utt_tok_label = [0 for _ in sys_utt_tok]
  usr_utt_tok_label = [0 for _ in usr_utt_tok]
  if label == 'none' or label == 'dontcare':
    class_type = label
  else:
    label, in_usr, usr_pos, in_sys, sys_pos = check_label_existence(label, usr_utt_tok, sys_utt_tok)
    if in_usr or in_sys:
      class_type = 'copy_value'
      if slot_last_occurrence:
        if in_usr:
          (s, e) = usr_pos[-1]
          for i in range(s, e):
            usr_utt_tok_label[i] = 1
        else:
          (s, e) = sys_pos[-1]
          for i in range(s, e):
            sys_utt_tok_label[i] = 1
      else:
        for (s, e) in usr_pos:
          for i in range(s, e):
            usr_utt_tok_label[i] = 1
        for (s, e) in sys_pos:
          for i in range(s, e):
            sys_utt_tok_label[i] = 1
    else:
      class_type = 'unpointable'
  return sys_utt_tok_label, usr_utt_tok_label, class_type


def tokenize(utt):
  utt_lower = tokenization.convert_to_unicode(utt).lower()
  utt_tok = [tok for tok in  map(str.strip, re.split("(\W+)", utt_lower)) if
             len(tok) > 0]
  return utt_tok


def create_examples(dialog_filename, slot_list, set_type, use_asr_hyp=0,
                    exclude_unpointable=True):
  examples = []
  with open(dialog_filename) as f:
    dst_set = json.load(f)
  for dial in dst_set:
    for turn in dial['dialogue']:
      guid = '%s-%s-%s' % (set_type,
                           str(dial['dialogue_idx']),
                           str(turn['turn_idx']))

      sys_utt_tok = tokenize(turn['system_transcript'])

      usr_utt_tok_list = []
      if use_asr_hyp == 0:
        usr_utt_tok_list.append(tokenize(turn['transcript']))
      else:
        for asr_hyp, _ in turn['asr'][:use_asr_hyp]:
          usr_utt_tok_list.append(tokenize(asr_hyp))

      turn_label = [[FIX.get(s.strip(), s.strip()), FIX.get(v.strip(), v.strip())] for s, v in turn['turn_label']]

      for usr_utt_tok in usr_utt_tok_list:
        sys_utt_tok_label_dict = {}
        usr_utt_tok_label_dict = {}
        class_type_dict = {}
        for slot in slot_list:
          label = 'none'
          for [s, v] in turn_label:
            if s == slot:
              label = v
              break
          sys_utt_tok_label, usr_utt_tok_label, class_type = get_turn_label(
            label, sys_utt_tok, usr_utt_tok,
            slot_last_occurrence=True)
          sys_utt_tok_label_dict[slot] = sys_utt_tok_label
          usr_utt_tok_label_dict[slot] = usr_utt_tok_label
          class_type_dict[slot] = class_type
          if class_type == 'unpointable':
            tf.logging.info(
              'Unpointable: guid=%s, slot=%s, label=%s, usr_utt=%s, sys_utt=%s' % (
              guid, slot, label, usr_utt_tok, sys_utt_tok))
        if 'unpointable' not in class_type_dict.values() or not exclude_unpointable:
          examples.append(util.InputExample(
            guid=guid,
            text_a=sys_utt_tok,
            text_b=usr_utt_tok,
            text_a_label=sys_utt_tok_label_dict,
            text_b_label=usr_utt_tok_label_dict,
            class_label=class_type_dict))
  return examples

def create_examples_with_history(dialog_filename, slot_list, set_type, use_asr_hyp=0, exclude_unpointable=True):
  examples = []
  with open(dialog_filename) as f:
    dst_set = json.load(f)
  for dial in dst_set:
    if use_asr_hyp == 0:
      his_utt_list = [[]]
    else:
      his_utt_list = [[] for _ in range(use_asr_hyp)]

    for turn in dial['dialogue']:
      guid = '%s-%s-%s' % (set_type, str(dial['dialogue_idx']), str(turn['turn_idx']))

      sys_utt_tok = tokenize(turn['system_transcript'])

      for his_utt in his_utt_list:
        his_utt.append(sys_utt_tok)

      usr_utt_tok_list = []
      if use_asr_hyp == 0:
        usr_utt_tok_list.append(tokenize(turn['transcript']))
      else:
        for asr_hyp in turn.asr[:use_asr_hyp]:
          usr_utt_tok_list.append(tokenize(asr_hyp))

      turn_label = [[FIX.get(s.strip(), s.strip()), FIX.get(v.strip(), v.strip())] for s, v in turn['turn_label']]

      for his_utt, usr_utt_tok in zip(his_utt_list, usr_utt_tok_list):
        his_utt_tok = []
        for utt_tok in his_utt[-5:]:
          his_utt_tok.extend(utt_tok)
        his_utt_tok_label_dict = {}
        usr_utt_tok_label_dict = {}
        class_type_dict = {}
        for slot in slot_list:
          label = 'none'
          for [s, v] in turn_label:
            if s == slot:
              label = v
              break
          his_utt_tok_label, usr_utt_tok_label, class_type = get_turn_label(
            label, his_utt_tok, usr_utt_tok, slot_last_occurrence=True)
          his_utt_tok_label_dict[slot] = his_utt_tok_label
          usr_utt_tok_label_dict[slot] = usr_utt_tok_label
          class_type_dict[slot] = class_type
          if class_type == 'unpointable':
            tf.logging.info('Unpointable: guid=%s, slot=%s, label=%s, his_utt=%s, usr_utt=%s' % (guid, slot, label, his_utt_tok, usr_utt_tok))
        if 'unpointable' not in class_type_dict.values() or not exclude_unpointable:
          examples.append(util.InputExample(guid=guid,
                                            text_a=his_utt_tok,
                                            text_b=usr_utt_tok,
                                            text_a_label=his_utt_tok_label_dict,
                                            text_b_label=usr_utt_tok_label_dict,
                                            class_label=class_type_dict))

      for his_utt, usr_utt_tok in zip(his_utt_list, usr_utt_tok_list):
        his_utt.append(usr_utt_tok)
  return examples


