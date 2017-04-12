# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nnquery.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nnquery.proto',
  package='nnxs',
  syntax='proto3',
  serialized_pb=_b('\n\rnnquery.proto\x12\x04nnxs\"B\n\x06NNRect\x12\x0c\n\x04left\x18\x01 \x01(\x05\x12\x0b\n\x03top\x18\x02 \x01(\x05\x12\r\n\x05right\x18\x03 \x01(\x05\x12\x0e\n\x06\x62ottom\x18\x04 \x01(\x05\"&\n\x07NNImage\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\r\n\x05\x63olor\x18\x02 \x01(\x08\"*\n\x12NNDescriptorResult\x12\x14\n\x08\x64\x65script\x18\x01 \x03(\x02\x42\x02\x10\x01\"\xb4\x01\n\x16NNClassificationResult\x12\x1c\n\x14most_likely_class_id\x18\x01 \x01(\x05\x12\x1e\n\x16most_likely_class_prob\x18\x02 \x01(\x02\x12\x1e\n\x16most_likely_class_name\x18\x03 \x01(\t\x12\x11\n\x05probs\x18\x04 \x03(\x02\x42\x02\x10\x01\x12\x15\n\tclass_ids\x18\x05 \x03(\x05\x42\x02\x10\x01\x12\x12\n\nerror_code\x18\x06 \x01(\x05\"\x88\x01\n\x11NNDetectionResult\x12\x1b\n\x05rects\x18\x01 \x03(\x0b\x32\x0c.nnxs.NNRect\x12.\n\x08\x63results\x18\x02 \x03(\x0b\x32\x1c.nnxs.NNClassificationResult\x12\x12\n\nobjectness\x18\x03 \x03(\x02\x12\x12\n\nerror_code\x18\x04 \x01(\x05\"f\n\tNNRequest\x12\r\n\x05reqid\x18\x01 \x01(\x03\x12\r\n\x05query\x18\x02 \x01(\t\x12\x1c\n\x05image\x18\x03 \x01(\x0b\x32\r.nnxs.NNImage\x12\x1d\n\x07windows\x18\x04 \x03(\x0b\x32\x0c.nnxs.NNRect\"\xe9\x01\n\x08NNResult\x12\r\n\x05reqid\x18\x01 \x01(\x03\x12\x12\n\nerror_code\x18\x02 \x01(\x05\x12\x15\n\rerror_message\x18\x03 \x01(\t\x12;\n\x15\x63lassification_result\x18\x07 \x03(\x0b\x32\x1c.nnxs.NNClassificationResult\x12\x31\n\x10\x64\x65tection_result\x18\x08 \x03(\x0b\x32\x17.nnxs.NNDetectionResult\x12\x33\n\x11\x64\x65scriptor_result\x18\t \x03(\x0b\x32\x18.nnxs.NNDescriptorResultb\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_NNRECT = _descriptor.Descriptor(
  name='NNRect',
  full_name='nnxs.NNRect',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='left', full_name='nnxs.NNRect.left', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='top', full_name='nnxs.NNRect.top', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='right', full_name='nnxs.NNRect.right', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='bottom', full_name='nnxs.NNRect.bottom', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=23,
  serialized_end=89,
)


_NNIMAGE = _descriptor.Descriptor(
  name='NNImage',
  full_name='nnxs.NNImage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='nnxs.NNImage.data', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='color', full_name='nnxs.NNImage.color', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=91,
  serialized_end=129,
)


_NNDESCRIPTORRESULT = _descriptor.Descriptor(
  name='NNDescriptorResult',
  full_name='nnxs.NNDescriptorResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='descript', full_name='nnxs.NNDescriptorResult.descript', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=131,
  serialized_end=173,
)


_NNCLASSIFICATIONRESULT = _descriptor.Descriptor(
  name='NNClassificationResult',
  full_name='nnxs.NNClassificationResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='most_likely_class_id', full_name='nnxs.NNClassificationResult.most_likely_class_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='most_likely_class_prob', full_name='nnxs.NNClassificationResult.most_likely_class_prob', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='most_likely_class_name', full_name='nnxs.NNClassificationResult.most_likely_class_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='probs', full_name='nnxs.NNClassificationResult.probs', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
    _descriptor.FieldDescriptor(
      name='class_ids', full_name='nnxs.NNClassificationResult.class_ids', index=4,
      number=5, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
    _descriptor.FieldDescriptor(
      name='error_code', full_name='nnxs.NNClassificationResult.error_code', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=176,
  serialized_end=356,
)


_NNDETECTIONRESULT = _descriptor.Descriptor(
  name='NNDetectionResult',
  full_name='nnxs.NNDetectionResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='rects', full_name='nnxs.NNDetectionResult.rects', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cresults', full_name='nnxs.NNDetectionResult.cresults', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='objectness', full_name='nnxs.NNDetectionResult.objectness', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='error_code', full_name='nnxs.NNDetectionResult.error_code', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=359,
  serialized_end=495,
)


_NNREQUEST = _descriptor.Descriptor(
  name='NNRequest',
  full_name='nnxs.NNRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='reqid', full_name='nnxs.NNRequest.reqid', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='query', full_name='nnxs.NNRequest.query', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image', full_name='nnxs.NNRequest.image', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='windows', full_name='nnxs.NNRequest.windows', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=497,
  serialized_end=599,
)


_NNRESULT = _descriptor.Descriptor(
  name='NNResult',
  full_name='nnxs.NNResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='reqid', full_name='nnxs.NNResult.reqid', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='error_code', full_name='nnxs.NNResult.error_code', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='error_message', full_name='nnxs.NNResult.error_message', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='classification_result', full_name='nnxs.NNResult.classification_result', index=3,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='detection_result', full_name='nnxs.NNResult.detection_result', index=4,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='descriptor_result', full_name='nnxs.NNResult.descriptor_result', index=5,
      number=9, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=602,
  serialized_end=835,
)

_NNDETECTIONRESULT.fields_by_name['rects'].message_type = _NNRECT
_NNDETECTIONRESULT.fields_by_name['cresults'].message_type = _NNCLASSIFICATIONRESULT
_NNREQUEST.fields_by_name['image'].message_type = _NNIMAGE
_NNREQUEST.fields_by_name['windows'].message_type = _NNRECT
_NNRESULT.fields_by_name['classification_result'].message_type = _NNCLASSIFICATIONRESULT
_NNRESULT.fields_by_name['detection_result'].message_type = _NNDETECTIONRESULT
_NNRESULT.fields_by_name['descriptor_result'].message_type = _NNDESCRIPTORRESULT
DESCRIPTOR.message_types_by_name['NNRect'] = _NNRECT
DESCRIPTOR.message_types_by_name['NNImage'] = _NNIMAGE
DESCRIPTOR.message_types_by_name['NNDescriptorResult'] = _NNDESCRIPTORRESULT
DESCRIPTOR.message_types_by_name['NNClassificationResult'] = _NNCLASSIFICATIONRESULT
DESCRIPTOR.message_types_by_name['NNDetectionResult'] = _NNDETECTIONRESULT
DESCRIPTOR.message_types_by_name['NNRequest'] = _NNREQUEST
DESCRIPTOR.message_types_by_name['NNResult'] = _NNRESULT

NNRect = _reflection.GeneratedProtocolMessageType('NNRect', (_message.Message,), dict(
  DESCRIPTOR = _NNRECT,
  __module__ = 'nnquery_pb2'
  # @@protoc_insertion_point(class_scope:nnxs.NNRect)
  ))
_sym_db.RegisterMessage(NNRect)

NNImage = _reflection.GeneratedProtocolMessageType('NNImage', (_message.Message,), dict(
  DESCRIPTOR = _NNIMAGE,
  __module__ = 'nnquery_pb2'
  # @@protoc_insertion_point(class_scope:nnxs.NNImage)
  ))
_sym_db.RegisterMessage(NNImage)

NNDescriptorResult = _reflection.GeneratedProtocolMessageType('NNDescriptorResult', (_message.Message,), dict(
  DESCRIPTOR = _NNDESCRIPTORRESULT,
  __module__ = 'nnquery_pb2'
  # @@protoc_insertion_point(class_scope:nnxs.NNDescriptorResult)
  ))
_sym_db.RegisterMessage(NNDescriptorResult)

NNClassificationResult = _reflection.GeneratedProtocolMessageType('NNClassificationResult', (_message.Message,), dict(
  DESCRIPTOR = _NNCLASSIFICATIONRESULT,
  __module__ = 'nnquery_pb2'
  # @@protoc_insertion_point(class_scope:nnxs.NNClassificationResult)
  ))
_sym_db.RegisterMessage(NNClassificationResult)

NNDetectionResult = _reflection.GeneratedProtocolMessageType('NNDetectionResult', (_message.Message,), dict(
  DESCRIPTOR = _NNDETECTIONRESULT,
  __module__ = 'nnquery_pb2'
  # @@protoc_insertion_point(class_scope:nnxs.NNDetectionResult)
  ))
_sym_db.RegisterMessage(NNDetectionResult)

NNRequest = _reflection.GeneratedProtocolMessageType('NNRequest', (_message.Message,), dict(
  DESCRIPTOR = _NNREQUEST,
  __module__ = 'nnquery_pb2'
  # @@protoc_insertion_point(class_scope:nnxs.NNRequest)
  ))
_sym_db.RegisterMessage(NNRequest)

NNResult = _reflection.GeneratedProtocolMessageType('NNResult', (_message.Message,), dict(
  DESCRIPTOR = _NNRESULT,
  __module__ = 'nnquery_pb2'
  # @@protoc_insertion_point(class_scope:nnxs.NNResult)
  ))
_sym_db.RegisterMessage(NNResult)


_NNDESCRIPTORRESULT.fields_by_name['descript'].has_options = True
_NNDESCRIPTORRESULT.fields_by_name['descript']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_NNCLASSIFICATIONRESULT.fields_by_name['probs'].has_options = True
_NNCLASSIFICATIONRESULT.fields_by_name['probs']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_NNCLASSIFICATIONRESULT.fields_by_name['class_ids'].has_options = True
_NNCLASSIFICATIONRESULT.fields_by_name['class_ids']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
# @@protoc_insertion_point(module_scope)
