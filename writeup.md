## BPE tokenizer

### unicode
- What Unicode character does chr(0) return?
  '\x00'

- How does this character’s string representation (__repr__()) differ from its printed representa-
tion?
  The printed representation is blank string, the __repr__ one is '\x00'

- What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:
```
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
It shows that: 
```
>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```