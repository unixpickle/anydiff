package anydifftest

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestReduceBatch(t *testing.T) {
	s := &anyseq.Batch{
		Packed: anyvec32.MakeVectorData([]float32{
			1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
		}),
		Present: []bool{true, false, true, true, false, false, true, true},
	}
	reduced := anyseq.ReduceBatch(s, []bool{true, false, false, true, false, false,
		false, true})
	expected := []float32{1, 2, 5, 6, 9, 10}
	actual := reduced.Packed.Data().([]float32)
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %v but got %v", expected, actual)
	}

	reduced = anyseq.ReduceBatch(s, []bool{false, false, true, false, false, false,
		true, false})
	expected = []float32{3, 4, 7, 8}
	actual = reduced.Packed.Data().([]float32)
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %v but got %v", expected, actual)
	}
}

func TestExpandBatch(t *testing.T) {
	s := &anyseq.Batch{
		Packed: anyvec32.MakeVectorData([]float32{
			1, 2, 3, 4, 5, 6,
		}),
		Present: []bool{true, false, true, false, false, false, false, true},
	}
	expanded := anyseq.ExpandBatch(s, []bool{true, false, true, false, true, false,
		true, true})
	expected := []float32{1, 2, 3, 4, 0, 0, 0, 0, 5, 6}
	actual := expanded.Packed.Data().([]float32)
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %v but got %v", expected, actual)
	}

	expanded = anyseq.ExpandBatch(s, []bool{true, true, true, true, true, true,
		true, true})
	expected = []float32{1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6}
	actual = expanded.Packed.Data().([]float32)
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %v but got %v", expected, actual)
	}

	expanded = anyseq.ExpandBatch(s, []bool{true, false, true, false, true, true,
		true, true})
	expected = []float32{1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 5, 6}
	actual = expanded.Packed.Data().([]float32)
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %v but got %v", expected, actual)
	}
}

func TestReduce(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		inSeq, varList := makeBasicTestSeqs(c)
		ch := &SeqChecker{
			F: func() anyseq.Seq {
				return anyseq.Reduce(inSeq, []bool{true, false, true, false})
			},
			V: varList,
		}
		ch.FullCheck(t)
	})
}
