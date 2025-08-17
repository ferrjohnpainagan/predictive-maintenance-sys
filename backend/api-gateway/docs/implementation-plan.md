# NestJS API Gateway Implementation Plan - Predictive Maintenance System

## Project Overview
Implementation plan for the NestJS API Gateway that serves as the central orchestrator for the predictive maintenance system. This gateway handles client requests, manages authentication via Supabase, coordinates between services, and provides a unified API interface.

## Technology Stack
- **Framework**: NestJS (Latest version)
- **Language**: TypeScript
- **Database**: Supabase (Postgres)
- **Authentication**: Supabase Auth (JWT)
- **Communication**: gRPC (for ML service), REST (for frontend)
- **Documentation**: OpenAPI/Swagger
- **Testing**: Jest, Supertest
- **Logging**: Pino
- **Deployment**: AWS Fargate/EC2 via Docker

---

## Phase 1: Project Setup and Foundation
**Duration**: 2-3 days  
**Priority**: Critical

### Objectives
- Initialize NestJS project with TypeScript
- Configure development environment
- Set up project structure and conventions

### Tasks
1. **Project Initialization**
   - [ ] Create NestJS project with CLI
   - [ ] Configure TypeScript (strict mode)
   - [ ] Set up ESLint and Prettier
   - [ ] Configure Git repository
   - [ ] Set up environment configuration

2. **Dependencies Installation**
   ```bash
   # Core dependencies
   npm install @nestjs/config @nestjs/swagger
   npm install @supabase/supabase-js
   npm install @grpc/grpc-js @grpc/proto-loader
   npm install pino pino-pretty nestjs-pino
   npm install class-validator class-transformer
   npm install @nestjs/throttler helmet
   ```

3. **Project Structure**
   ```
   backend/api-gateway/
   ├── src/
   │   ├── auth/                   # Authentication module
   │   ├── engines/                # Engine management module
   │   ├── fleet/                  # Fleet overview module
   │   ├── predictions/            # ML predictions module
   │   ├── common/                 # Shared resources
   │   │   ├── decorators/
   │   │   ├── guards/
   │   │   ├── interceptors/
   │   │   ├── filters/
   │   │   └── pipes/
   │   ├── config/                 # Configuration module
   │   ├── grpc/                   # gRPC client module
   │   ├── supabase/              # Supabase integration
   │   └── main.ts
   ├── proto/                      # Protocol buffer definitions
   ├── test/                       # Test files
   └── docker/                     # Docker configuration
   ```

4. **Configuration Setup**
   ```typescript
   // config/configuration.ts
   export default () => ({
     port: parseInt(process.env.PORT, 10) || 3000,
     supabase: {
       url: process.env.SUPABASE_URL,
       anonKey: process.env.SUPABASE_ANON_KEY,
       serviceKey: process.env.SUPABASE_SERVICE_KEY,
     },
     mlService: {
       host: process.env.ML_SERVICE_HOST || 'localhost',
       port: parseInt(process.env.ML_SERVICE_PORT, 10) || 50051,
     },
     cors: {
       origin: process.env.FRONTEND_URL || 'http://localhost:3001',
     },
   });
   ```

### Deliverables
- Initialized NestJS project
- Configured development environment
- Basic project structure

### Testing Checklist
- [ ] Project builds successfully
- [ ] Development server runs
- [ ] TypeScript compilation passes
- [ ] Environment variables load correctly

---

## Phase 2: Core Infrastructure and Middleware
**Duration**: 3-4 days  
**Priority**: High

### Objectives
- Set up core infrastructure components
- Implement logging and monitoring
- Configure security middleware

### Tasks
1. **Logging System**
   ```typescript
   // common/logging/logging.module.ts
   import { LoggerModule } from 'nestjs-pino';
   
   @Module({
     imports: [
       LoggerModule.forRoot({
         pinoHttp: {
           customProps: (req) => ({
             correlationId: req.headers['x-correlation-id'],
           }),
           transport: {
             target: 'pino-pretty',
           },
         },
       }),
     ],
   })
   export class LoggingModule {}
   ```

2. **Exception Handling**
   - [ ] Global exception filter
   - [ ] HTTP exception filter
   - [ ] gRPC exception handling
   - [ ] Validation pipe setup

3. **Security Middleware**
   - [ ] Helmet integration
   - [ ] Rate limiting (throttler)
   - [ ] CORS configuration
   - [ ] Request validation

4. **Health Checks**
   ```typescript
   // health/health.controller.ts
   @Controller('health')
   export class HealthController {
     @Get()
     check() {
       return { status: 'ok', timestamp: new Date().toISOString() };
     }
     
     @Get('ready')
     async readiness() {
       // Check Supabase connection
       // Check ML service availability
       return { ready: true };
     }
   }
   ```

### Deliverables
- Configured logging system
- Security middleware setup
- Health check endpoints

### Testing Checklist
- [ ] Logs include correlation IDs
- [ ] Rate limiting works
- [ ] Health endpoints respond
- [ ] Exception handling works

---

## Phase 3: Supabase Integration
**Duration**: 4-5 days  
**Priority**: Critical

### Objectives
- Integrate with Supabase services
- Implement authentication
- Set up database operations

### Tasks
1. **Supabase Service Module**
   ```typescript
   // supabase/supabase.service.ts
   @Injectable()
   export class SupabaseService {
     private supabase: SupabaseClient;
     
     constructor(private config: ConfigService) {
       this.supabase = createClient(
         config.get('supabase.url'),
         config.get('supabase.serviceKey'),
       );
     }
     
     async getEngines() {
       const { data, error } = await this.supabase
         .from('engines')
         .select('*')
         .order('unit_number');
       return data;
     }
     
     async getEngineSensorData(engineId: string, limit = 50) {
       const { data, error } = await this.supabase
         .from('sensor_data')
         .select('*')
         .eq('engine_id', engineId)
         .order('cycle', { ascending: false })
         .limit(limit);
       return data;
     }
   }
   ```

2. **Authentication Guard**
   ```typescript
   // auth/guards/jwt-auth.guard.ts
   @Injectable()
   export class JwtAuthGuard implements CanActivate {
     constructor(private supabase: SupabaseService) {}
     
     async canActivate(context: ExecutionContext): Promise<boolean> {
       const request = context.switchToHttp().getRequest();
       const token = this.extractToken(request);
       
       if (!token) return false;
       
       const { data: { user }, error } = await this.supabase
         .auth.getUser(token);
       
       if (error || !user) return false;
       
       request.user = user;
       return true;
     }
   }
   ```

3. **Database Schema Types**
   ```typescript
   // types/database.types.ts
   export interface Engine {
     id: string;
     unit_number: string;
     cycles: number;
     last_maintenance: Date;
     status: 'operational' | 'maintenance' | 'retired';
   }
   
   export interface SensorData {
     id: string;
     engine_id: string;
     cycle: number;
     timestamp: Date;
     s2: number;  // T24
     s3: number;  // T30
     s4: number;  // T50
     s7: number;  // P30
     // ... other sensors
   }
   ```

4. **Storage Integration**
   - [ ] File upload service
   - [ ] Dataset retrieval
   - [ ] Model artifact management

### Deliverables
- Complete Supabase integration
- Working authentication
- Database operations

### Testing Checklist
- [ ] Authentication validates JWTs
- [ ] Database queries work
- [ ] Storage operations function
- [ ] Error handling for Supabase

---

## Phase 4: gRPC Client for ML Service
**Duration**: 3-4 days  
**Priority**: High

### Objectives
- Set up gRPC client
- Define protocol buffers
- Implement ML service communication

### Tasks
1. **Protocol Buffer Definition**
   ```protobuf
   // proto/prediction.proto
   syntax = "proto3";
   
   package prediction;
   
   service PredictionService {
     rpc PredictRUL (RULRequest) returns (RULResponse);
     rpc BatchPredictRUL (BatchRULRequest) returns (BatchRULResponse);
   }
   
   message SensorReading {
     float s2 = 1;
     float s3 = 2;
     float s4 = 3;
     float s7 = 4;
     float s8 = 5;
     float s9 = 6;
     float s11 = 7;
     float s12 = 8;
     float s13 = 9;
     float s14 = 10;
     float s15 = 11;
     float s17 = 12;
     float s20 = 13;
     float s21 = 14;
   }
   
   message RULRequest {
     string engine_id = 1;
     repeated SensorReading sensor_data = 2;
   }
   
   message RULResponse {
     int32 predicted_rul = 1;
     float confidence = 2;
     string model_version = 3;
   }
   ```

2. **gRPC Client Service**
   ```typescript
   // grpc/ml-client.service.ts
   @Injectable()
   export class MLClientService implements OnModuleInit {
     private client: any;
     
     onModuleInit() {
       const packageDefinition = protoLoader.loadSync(
         'proto/prediction.proto',
         {
           keepCase: true,
           longs: String,
           enums: String,
           defaults: true,
           oneofs: true,
         },
       );
       
       const proto = grpc.loadPackageDefinition(packageDefinition);
       this.client = new proto.prediction.PredictionService(
         `${this.config.get('mlService.host')}:${this.config.get('mlService.port')}`,
         grpc.credentials.createInsecure(),
       );
     }
     
     async predictRUL(engineId: string, sensorData: any[]): Promise<number> {
       return new Promise((resolve, reject) => {
         this.client.PredictRUL(
           { engine_id: engineId, sensor_data: sensorData },
           (error, response) => {
             if (error) reject(error);
             else resolve(response.predicted_rul);
           },
         );
       });
     }
   }
   ```

3. **Connection Management**
   - [ ] Connection pooling
   - [ ] Retry logic
   - [ ] Circuit breaker pattern
   - [ ] Timeout handling

### Deliverables
- gRPC client implementation
- Protocol buffer definitions
- ML service communication

### Testing Checklist
- [ ] gRPC client connects
- [ ] Predictions return correctly
- [ ] Error handling works
- [ ] Timeout handling functions

---

## Phase 5: API Endpoints Implementation
**Duration**: 5-6 days  
**Priority**: Critical

### Objectives
- Implement all REST API endpoints
- Create DTOs and validation
- Document with OpenAPI

### Tasks
1. **Fleet Module**
   ```typescript
   // fleet/fleet.controller.ts
   @Controller('api/fleet')
   @UseGuards(JwtAuthGuard)
   export class FleetController {
     constructor(
       private supabase: SupabaseService,
       private mlClient: MLClientService,
     ) {}
     
     @Get()
     async getFleetSummary() {
       const engines = await this.supabase.getEngines();
       
       // Parallel RUL predictions
       const predictions = await Promise.all(
         engines.map(async (engine) => {
           const sensorData = await this.supabase
             .getEngineSensorData(engine.id);
           const rul = await this.mlClient
             .predictRUL(engine.id, sensorData);
           return { ...engine, predicted_rul: rul };
         }),
       );
       
       return {
         engines: predictions,
         summary: {
           total: engines.length,
           critical: predictions.filter(e => e.predicted_rul < 20).length,
           warning: predictions.filter(e => e.predicted_rul < 50).length,
           average_rul: predictions.reduce((acc, e) => acc + e.predicted_rul, 0) / predictions.length,
         },
       };
     }
   }
   ```

2. **Engine Module**
   ```typescript
   // engines/engines.controller.ts
   @Controller('api/engines')
   @UseGuards(JwtAuthGuard)
   export class EnginesController {
     @Get(':id')
     async getEngineDetail(@Param('id') id: string) {
       const engine = await this.supabase.getEngine(id);
       const sensorHistory = await this.supabase
         .getEngineSensorData(id, 500);
       
       return {
         engine,
         sensor_history: sensorHistory,
         statistics: this.calculateStatistics(sensorHistory),
       };
     }
     
     @Post(':id/predict')
     async predictRUL(
       @Param('id') id: string,
       @Body() dto: PredictionRequestDto,
     ) {
       const sensorData = dto.sensor_data || 
         await this.supabase.getEngineSensorData(id);
       
       const rul = await this.mlClient.predictRUL(id, sensorData);
       
       // Store prediction in database
       await this.supabase.storePrediction({
         engine_id: id,
         predicted_rul: rul,
         timestamp: new Date(),
       });
       
       return { 
         engine_id: id,
         predicted_rul: rul,
         timestamp: new Date(),
       };
     }
   }
   ```

3. **DTOs and Validation**
   ```typescript
   // dto/prediction.dto.ts
   export class PredictionRequestDto {
     @IsOptional()
     @IsArray()
     sensor_data?: SensorReadingDto[];
   }
   
   export class SensorReadingDto {
     @IsNumber()
     s2: number;
     
     @IsNumber()
     s3: number;
     
     // ... other sensors
   }
   
   export class PredictionResponseDto {
     engine_id: string;
     predicted_rul: number;
     confidence?: number;
     timestamp: Date;
   }
   ```

4. **OpenAPI Documentation**
   ```typescript
   // main.ts
   const config = new DocumentBuilder()
     .setTitle('Predictive Maintenance API')
     .setDescription('API for aircraft engine health monitoring')
     .setVersion('1.0')
     .addBearerAuth()
     .build();
   
   const document = SwaggerModule.createDocument(app, config);
   SwaggerModule.setup('api/docs', app, document);
   ```

### Deliverables
- All API endpoints implemented
- DTOs with validation
- OpenAPI documentation

### Testing Checklist
- [ ] All endpoints return correct data
- [ ] Validation rejects invalid input
- [ ] OpenAPI docs accessible
- [ ] Authentication enforced

---

## Phase 6: Performance Optimization
**Duration**: 3-4 days  
**Priority**: High

### Objectives
- Optimize API performance
- Implement caching
- Add monitoring

### Tasks
1. **Caching Strategy**
   ```typescript
   // common/cache/cache.service.ts
   @Injectable()
   export class CacheService {
     private cache = new Map<string, any>();
     
     async get<T>(key: string, factory: () => Promise<T>, ttl = 60000): Promise<T> {
       const cached = this.cache.get(key);
       if (cached && cached.expires > Date.now()) {
         return cached.value;
       }
       
       const value = await factory();
       this.cache.set(key, {
         value,
         expires: Date.now() + ttl,
       });
       
       return value;
     }
   }
   ```

2. **Query Optimization**
   - [ ] Batch database queries
   - [ ] Implement pagination
   - [ ] Add database indexes
   - [ ] Connection pooling

3. **Response Optimization**
   - [ ] Response compression
   - [ ] Field filtering
   - [ ] Partial responses
   - [ ] ETags support

4. **Monitoring**
   ```typescript
   // common/monitoring/metrics.interceptor.ts
   @Injectable()
   export class MetricsInterceptor implements NestInterceptor {
     intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
       const start = Date.now();
       const request = context.switchToHttp().getRequest();
       
       return next.handle().pipe(
         tap(() => {
           const duration = Date.now() - start;
           this.logger.log({
             method: request.method,
             url: request.url,
             duration,
             status: context.switchToHttp().getResponse().statusCode,
           });
         }),
       );
     }
   }
   ```

### Deliverables
- Caching implementation
- Optimized queries
- Performance monitoring

### Testing Checklist
- [ ] API response < 200ms P95
- [ ] Caching reduces load
- [ ] Monitoring captures metrics
- [ ] No memory leaks

---

## Phase 7: Testing and Quality Assurance
**Duration**: 4-5 days  
**Priority**: High

### Objectives
- Implement comprehensive testing
- Set up CI/CD pipeline
- Ensure code quality

### Tasks
1. **Unit Testing**
   ```typescript
   // engines/engines.service.spec.ts
   describe('EnginesService', () => {
     let service: EnginesService;
     let supabase: MockSupabaseService;
     
     beforeEach(async () => {
       const module = await Test.createTestingModule({
         providers: [
           EnginesService,
           {
             provide: SupabaseService,
             useClass: MockSupabaseService,
           },
         ],
       }).compile();
       
       service = module.get<EnginesService>(EnginesService);
     });
     
     it('should return engine details', async () => {
       const result = await service.getEngineDetail('engine-1');
       expect(result).toBeDefined();
       expect(result.id).toBe('engine-1');
     });
   });
   ```

2. **Integration Testing**
   ```typescript
   // test/fleet.e2e-spec.ts
   describe('Fleet Controller (e2e)', () => {
     let app: INestApplication;
     
     beforeAll(async () => {
       const moduleFixture = await Test.createTestingModule({
         imports: [AppModule],
       }).compile();
       
       app = moduleFixture.createNestApplication();
       await app.init();
     });
     
     it('/api/fleet (GET)', () => {
       return request(app.getHttpServer())
         .get('/api/fleet')
         .set('Authorization', 'Bearer valid-token')
         .expect(200)
         .expect((res) => {
           expect(res.body.engines).toBeDefined();
           expect(res.body.summary).toBeDefined();
         });
     });
   });
   ```

3. **Load Testing**
   - [ ] Set up k6 or Artillery
   - [ ] Test concurrent users
   - [ ] Identify bottlenecks
   - [ ] Stress test endpoints

4. **Code Quality**
   - [ ] ESLint rules enforcement
   - [ ] Code coverage > 80%
   - [ ] SonarQube integration
   - [ ] Security scanning

### Deliverables
- Complete test suite
- Load testing results
- Code quality reports

### Testing Checklist
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Load tests meet targets
- [ ] Code coverage > 80%

---

## Phase 8: Deployment and DevOps
**Duration**: 4-5 days  
**Priority**: Critical

### Objectives
- Containerize application
- Set up CI/CD pipeline
- Deploy to AWS

### Tasks
1. **Docker Configuration**
   - Refer to **Infrastructure as Code Implementation Plan** for complete containerization setup
   - NestJS-specific Docker optimizations
   - Multi-stage builds for production

2. **CI/CD Pipeline**
   - See **Infrastructure as Code Implementation Plan** for comprehensive CI/CD setup
   - Node.js specific testing and build steps
   - Integration with GitHub Actions workflow

3. **AWS Infrastructure**
   - Refer to **Infrastructure as Code Implementation Plan** for complete AWS setup:
     - ECR repository configuration
     - ECS Fargate task definitions
     - Application Load Balancer setup
     - Auto-scaling policies
     - CloudWatch logging integration

4. **Monitoring Integration**
   - See **Monitoring & Observability Implementation Plan** for:
     - Application performance monitoring
     - Distributed tracing setup
     - Health check endpoints
     - Metrics collection

### Deliverables
- Docker configuration (see Infrastructure plan for details)
- CI/CD pipeline (see Infrastructure plan)
- AWS deployment (see Infrastructure plan)
- Monitoring setup (see Monitoring plan)

### Testing Checklist
- [ ] Docker image builds (Infrastructure plan)
- [ ] CI/CD pipeline works (Infrastructure plan)
- [ ] Application deploys (Infrastructure plan)
- [ ] Health checks pass
- [ ] Monitoring active (Monitoring plan)

---

## Success Metrics

### Performance KPIs
- API response time: < 200ms (P95)
- Throughput: > 1000 req/sec
- Error rate: < 0.1%
- Availability: > 99.9%

### Code Quality
- Test coverage: > 80%
- No critical security issues
- Documentation complete
- All endpoints tested

---

## Risk Mitigation

### Technical Risks
1. **Supabase Downtime**
   - Mitigation: Implement circuit breaker and fallback mechanisms

2. **ML Service Latency**
   - Mitigation: Implement caching and batch processing

3. **High Traffic Load**
   - Mitigation: Auto-scaling and load balancing

### Dependencies
1. **ML Service Availability**
   - Mitigation: Mock service for development

2. **Supabase Rate Limits**
   - Mitigation: Implement request batching and caching

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Setup | 2-3 days | None |
| Phase 2: Infrastructure | 3-4 days | Phase 1 |
| Phase 3: Supabase | 4-5 days | Phase 2 |
| Phase 4: gRPC Client | 3-4 days | Phase 2 |
| Phase 5: API Endpoints | 5-6 days | Phase 3, 4 |
| Phase 6: Optimization | 3-4 days | Phase 5 |
| Phase 7: Testing | 4-5 days | Phase 5 |
| Phase 8: Deployment | 4-5 days | Phase 7 |

**Total Duration**: 28-36 days (5-7 weeks)

---

## Next Steps
1. Review and approve implementation plan
2. Set up development environment
3. Begin Phase 1 implementation
4. Coordinate with ML team for gRPC interface
5. Coordinate with frontend team for API specifications