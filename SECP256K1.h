#ifndef SECP256K1_H
#define SECP256K1_H

#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <stdio.h>
#include "Int.h"
#include "Point.h"
#include "IntGroup.h"

// Secp256K1 Elliptic Curve
// y^2 = x^3 + 7
// Finite field: FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// Order:        FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// Generator:    G=(79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
//                  483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)

class Secp256K1 {

public:

  Secp256K1();
  void Init();
  Point ComputePublicKey(Int *privKey);
  void Check();
  Int GetY(Int x, bool isEven);
  bool EC(Int &x, Int &y);
  
  // Dodawanie i podwajanie punktów
  Point AddDirect(Point &p1, Point &p2);
  Point AddDirectAVX512(Point &p1, Point &p2);
  Point DoubleDirect(Point &p);
  Point DoubleDirectAVX512(Point &p);
  Point AddJacobian(Point &p1, Point &p2);
  Point DoubleJacobian(Point &p);
  
  // Operacje na punktach
  Point NextKey(Point &key);
  bool PointAtInfinity(Point &p);
  void PrefetchPoint(const Point& p, int hint);
  
  // Mnożenie skalarne
  Point ScalarMultiplication(Point &p, Int *scalar, bool isBatchMode = false);
  
  // Operacje na podpisach
  bool VerifySignature(Int &hash, Int &r, Int &s, Point &pubKey);
  bool BatchVerify(int batchSize, Point *publicKeys, Int *hashes, Int *rs, Int *ss);
  void SNARK_Proof(Int &x, Int &y, Int &r);
  
  // Kompresja i dekompresja punktów
  Point CompressPoint(Point &p);
  Point DecompressPoint(Point &compressedPoint);

  // Pola publiczne
  Int P;      // Characteristic of the finite field
  Int order;  // Order of the generator point
  Point G;    // Generator point
};

#endif // SECP256K1_H
