// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package arrow

type EncodedType interface {
	DataType
	Encoded() DataType
}

// RunEndEncodedType is the datatype to represent a run-end encoded
// array of data.
type RunEndEncodedType struct {
	ends DataType
	enc  DataType
}

func RunEndEncodedOf(runEnds, encoded DataType) *RunEndEncodedType {
	return &RunEndEncodedType{ends: runEnds, enc: encoded}
}

func (*RunEndEncodedType) ID() Type     { return RUN_END_ENCODED }
func (*RunEndEncodedType) Name() string { return "run_end_encoded" }
func (*RunEndEncodedType) Layout() DataTypeLayout {
	return DataTypeLayout{Buffers: []BufferSpec{SpecAlwaysNull()}}
}

func (t *RunEndEncodedType) String() string {
	return t.Name() + "<run_ends: " + t.ends.String() + ", values: " + t.enc.String() + ">"
}

func (t *RunEndEncodedType) Fingerprint() string {
	return typeFingerprint(t) + "{" + t.ends.Fingerprint() + ";" + t.enc.Fingerprint() + ";}"
}

func (t *RunEndEncodedType) Encoded() DataType { return t.enc }

func (t *RunEndEncodedType) Fields() []Field {
	return []Field{
		{Name: "run_ends", Type: t.ends},
		{Name: "values", Type: t.enc, Nullable: true},
	}
}

func (*RunEndEncodedType) ValidRunEndsType(dt DataType) bool {
	switch dt.ID() {
	case INT16, INT32, INT64:
		return true
	}
	return false
}
