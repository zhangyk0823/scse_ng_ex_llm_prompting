from parse_data import load_items, get_unclaimed_items, save_result
from investigate import build_prompt, parse_response, validate_result, display_matches
import os

items = load_items('found_items.json')
print('total items:', len(items))
unclaimed = get_unclaimed_items(items)
print('unclaimed ids:', [i['id'] for i in unclaimed])

sp, up = build_prompt('I lost a black backpack in the library', unclaimed)
print('--- system prompt head ---')
print(sp[:200])
print('--- user prompt tail ---')
print(up[-200:])

# Test parse_response with fenced JSON
resp = 'Here is my answer:\n```json\n{"matches": ["F101"], "confidence": "MEDIUM"}\n```'
result = parse_response(resp)
print('parsed:', result)
print('valid:', validate_result(result, unclaimed))

# Test invalid confidence
bad = {'matches': ['F101'], 'confidence': 'SURE'}
print('bad conf valid:', validate_result(bad, unclaimed))

# Test invalid id
bad2 = {'matches': ['F999'], 'confidence': 'LOW'}
print('bad id valid:', validate_result(bad2, unclaimed))

# Test empty matches
empty = {'matches': [], 'confidence': 'LOW'}
print('empty valid:', validate_result(empty, unclaimed))

print('--- display matches ---')
display_matches(result, unclaimed)
print('--- display empty ---')
display_matches(empty, unclaimed)

save_result(result, 'output/match_result.json')
print('saved exists:', os.path.exists('output/match_result.json'))
