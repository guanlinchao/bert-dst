import json

import util


def dialogue_state_to_sv_dict(sv_list):
  sv_dict = {}
  for d in sv_list:
    sv_dict[d['slot']] = d['value']
  return sv_dict


def get_token_and_slot_label(turn):
  if 'system_utterance' in turn:
    sys_utt_tok = turn['system_utterance']['tokens']
    sys_slot_label = turn['system_utterance']['slots']
  else:
    sys_utt_tok = []
    sys_slot_label = []

  usr_utt_tok = turn['user_utterance']['tokens']
  usr_slot_label = turn['user_utterance']['slots']
  return sys_utt_tok, sys_slot_label, usr_utt_tok, usr_slot_label


def get_tok_label(prev_ds_dict, cur_ds_dict, slot_type, sys_utt_tok,
                  sys_slot_label, usr_utt_tok, usr_slot_label, dial_id,
                  turn_id, slot_last_occurrence=True):
  """The position of the last occurrence of the slot value will be used."""
  sys_utt_tok_label = [0 for _ in sys_utt_tok]
  usr_utt_tok_label = [0 for _ in usr_utt_tok]
  if slot_type not in cur_ds_dict:
    class_type = 'none'
  else:
    value = cur_ds_dict[slot_type]
    if value == 'dontcare' and (slot_type not in prev_ds_dict or
                                prev_ds_dict[slot_type] != 'dontcare'):
      # Only label dontcare at its first occurrence in the dialog
      class_type = 'dontcare'
    else: # If not none or dontcare, we have to identify the position
      class_type = 'copy_value'
      found_pos = False
      for label_d in usr_slot_label:
        if label_d['slot'] == slot_type and value == ' '.join(
            usr_utt_tok[label_d['start']:label_d['exclusive_end']]):

          for idx in range(label_d['start'], label_d['exclusive_end']):
            usr_utt_tok_label[idx] = 1
          found_pos = True
          if slot_last_occurrence:
            break
      if not found_pos or not slot_last_occurrence:
        for label_d in sys_slot_label:
          if label_d['slot'] == slot_type and value == ' '.join(
              sys_utt_tok[label_d['start']:label_d['exclusive_end']]):
            for idx in range(label_d['start'], label_d['exclusive_end']):
              sys_utt_tok_label[idx] = 1
            found_pos = True
            if slot_last_occurrence:
              break
      if not found_pos:
        assert sum(usr_utt_tok_label + sys_utt_tok_label) == 0
        if (slot_type not in prev_ds_dict or value != prev_ds_dict[slot_type]):
          raise ValueError('Copy value cannot found in Dial %s Turn %s' %
                           (str(dial_id), str(turn_id)))
        else:
          class_type = 'none'
      else:
        assert sum(usr_utt_tok_label + sys_utt_tok_label) > 0
  return sys_utt_tok_label, usr_utt_tok_label, class_type


def get_turn_label(turn, prev_dialogue_state, slot_list, dial_id, turn_id,
                   slot_last_occurrence=True):
  """Make turn_label a dictionary of slot with value positions or being dontcare / none:
    Turn label contains:
      (1) the updates from previous to current dialogue state,
      (2) values in current dialogue state explicitly mentioned in system or user utterance."""
  prev_ds_dict = dialogue_state_to_sv_dict(prev_dialogue_state)
  cur_ds_dict = dialogue_state_to_sv_dict(turn['dialogue_state'])

  (sys_utt_tok, sys_slot_label,
   usr_utt_tok, usr_slot_label) = get_token_and_slot_label(turn)

  sys_utt_tok_label_dict = {}
  usr_utt_tok_label_dict = {}
  class_type_dict = {}

  for slot_type in slot_list:
    sys_utt_tok_label, usr_utt_tok_label, class_type = get_tok_label(
      prev_ds_dict, cur_ds_dict, slot_type, sys_utt_tok, sys_slot_label,
      usr_utt_tok, usr_slot_label, dial_id, turn_id,
      slot_last_occurrence=slot_last_occurrence)
    sys_utt_tok_label_dict[slot_type] = sys_utt_tok_label
    usr_utt_tok_label_dict[slot_type] = usr_utt_tok_label
    class_type_dict[slot_type] = class_type
  return (sys_utt_tok, sys_utt_tok_label_dict,
          usr_utt_tok, usr_utt_tok_label_dict, class_type_dict)


def create_examples(dialog_filename, slot_list, set_type):
  examples = []
  with open(dialog_filename) as f:
    dst_set = json.load(f)
  for dial in dst_set:
    dial_id = dial['dialogue_id']
    prev_ds = []
    for turn_id, turn in enumerate(dial['turns']):
      guid = '%s-%s-%s' % (set_type, dial_id, str(turn_id))
      (sys_utt_tok,
       sys_utt_tok_label_dict,
       usr_utt_tok,
       usr_utt_tok_label_dict,
       class_type_dict) = get_turn_label(turn,
                                         prev_ds,
                                         slot_list,
                                         dial_id,
                                         turn_id,
                                         slot_last_occurrence=True)
      examples.append(util.InputExample(
        guid=guid,
        text_a=sys_utt_tok,
        text_b=usr_utt_tok,
        text_a_label=sys_utt_tok_label_dict,
        text_b_label=usr_utt_tok_label_dict,
        class_label=class_type_dict))
      prev_ds = turn['dialogue_state']
  return examples