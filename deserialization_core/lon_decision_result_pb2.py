# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lon_decision_result.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import lon_decision_obj_info_pb2 as lon__decision__obj__info__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x19lon_decision_result.proto\x12\x12lon_decision_proto\x1a\x1blon_decision_obj_info.proto\"_\n\x11ObjDecisionResult\x12\x13\n\x0borig_obj_id\x18\x01 \x01(\x05\x12\x35\n\x08\x64\x65\x63ision\x18\x02 \x01(\x0e\x32#.lon_decision_proto.LonDecisionType\"W\n\x11LonDecisionResult\x12\x42\n\x13obj_decision_result\x18\x01 \x03(\x0b\x32%.lon_decision_proto.ObjDecisionResultb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'lon_decision_result_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _OBJDECISIONRESULT._serialized_start=78
  _OBJDECISIONRESULT._serialized_end=173
  _LONDECISIONRESULT._serialized_start=175
  _LONDECISIONRESULT._serialized_end=262
# @@protoc_insertion_point(module_scope)
