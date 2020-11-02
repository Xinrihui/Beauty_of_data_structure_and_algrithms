
class InvertedIndex:

    def docID(self,node):
        pass


    def Intersect(self, p1, p2):
        answer=[]
        while p1!=None and p2!=None:
            if self.docID(p1) == self.docID(p2):
                answer.append(self.docID(p1))
                p1=p1.next()
                p2=p2.next()
            elif self.docID(p1) < self.docID(p2):
                p1 = p1.next()
            elif self.docID(p1) > self.docID(p2):
                p2 = p2.next()
        return answer

    def hasskip(self,p):
        pass

    def skip(self,p):
        pass

    def IntersectWithSkip(self,p1,p2):
        answer=[]
        while p1!=None and p2!=None:
            if self.docID(p1) == self.docID(p2):
                answer.append(self.docID(p1))
                p1=p1.next()
                p2=p2.next()

            elif  self.docID(p1) < self.docID(p2):
                while (self.hasskip(p1) and self.docID( self.skip(p1))<=self.docID(p2)):
                    p1=self.skip(p1)

                if self.docID(p1) == self.docID(p2): break
                else:
                    p1=p1.next()

            elif self.docID(p1) > self.docID(p2):
                while (self.hasskip(p2) and self.docID( self.skip(p2))<=self.docID(p1)):
                    p1=self.skip(p2)

                if self.docID(p1) == self.docID(p2): break
                else:
                    p2=p2.next()
        return answer



    def PositionalIntersect(self,p1,p2,k):
        """
        p1.positions -> [ 1 ,6 ,7 ]
        p2.positions -> [ 2,3,4,5,10 ]
        k=2
        pp1.pos=1   pp2.pos: [min=-1 ,max=3 ] （如果 p2.positions是有序的数组可以使用二分查找）匹配结果 ：<1,2> <1,3>
        pp1.pos=6   pp2.pos: [min=4 ,max=8 ]   匹配结果 ：<6,4> <6,5>
        pp1.pos=7   pp2.pos: [min=5 ,max=9 ]  匹配结果 ：<7,5> 
        :param p1: 
        :param p2: 
        :param k: 
        :return: 
        """
        answer=[]
        while p1!=None and p2!=None:
            if self.docID(p1) == self.docID(p2):
                l=[]
                pp1=p1.positions
                pp2=p2.positions

                while pp1!=None:
                    while pp2!=None:
                        if abs(pp2.pos - pp1.pos)<=k: #找到一个 pp2继续向右移动
                            l.append(pp2.pos)
                        elif (pp2.pos - pp1.pos)>k: # 如果pp2 已经到了区间的右边，不用再找了
                            break
                        elif (pp2.pos - pp1.pos)<k:continue # 如果pp2 还在区间的左边，需要继续向右边找

                        pp2=pp2.next

                    if len(l)>0: #匹配结果非空，才写入结果集
                        for ps in l:
                            answer.append([self.docID(p1),pp1.pos,ps]) #<文档 ID，词项在 p1中的位置，词项在 p2中的位置>
                    pp1=pp1.next

                p1=p1.next()
                p2=p2.next()
            elif self.docID(p1) < self.docID(p2):
                p1 = p1.next()
            elif self.docID(p1) > self.docID(p2):
                p2 = p2.next()
        return answer


